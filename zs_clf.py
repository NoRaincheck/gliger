"""JEV (Choice / Noul / Score) adapter for HuggingFace zero-shot classification pipeline.

This module provides adapters that emulate the TypeSafe API primitives (Choice, Noul, Score)
using HuggingFace's ``pipeline("zero-shot-classification")`` (e.g. NLI models like
``tasksource/ModernBERT-base-nli``). It follows the Typesafe API contract:

- Choice  -> multi-class probabilities (per-group renormalized softmax)
- Noul    -> binary yes/no probability (0=no, 1=yes)
- Score   -> ordered categorical (0..N with legend + probabilities)

The module uses a single HuggingFace pipeline call per text with all candidate labels
in one pass, enabling efficient batched processing.

See: https://docs.typesafe.ai/introduction/quickstart


Durable Design Decisions
=========================

1. **Single-pipeline call per text**
   All labels are passed as ``candidate_labels`` in one call. The NLI model evaluates
   every (text, label) pair independently, returning scores for all labels at once.

2. **Per-group renormalization**
   Raw NLI entailment scores are split by group and renormalized with softmax
   *within* each group, producing properly calibrated per-question probabilities.

3. **Score: fractional ordinal via reciprocal-rank weighting**
   Converts a probability distribution over ordered levels into a single fractional
   score in ``[0, N-1]`` — the most probable level (rank 1) gets weight 1, the next
   (rank 2) gets weight 1/2, etc., each multiplied by its probability.

4. **Noul: deterministic threshold at 0.5**
   Binary yes/no uses the raw ``yes`` probability from the group-renormalized softmax.
   ``noul = 1.0`` when ``yes_prob >= 0.5``, otherwise ``0.0``. Confidence is the max
   of the two class probabilities.

5. **Hierarchical label flattening**
   Converts ``{group: [labels]}`` into a flat list of dot-notation strings
   (``"group.label"``) and a reverse mapping ``flat_label -> group_name``.

6. **Unified prompt construction**
   All question instructions are joined with ``" | "`` into a single prompt string
   prepended to the text, shared by every text/question combination.

"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, pipeline


MODEL_ID: str = "tasksource/ModernBERT-base-nli"

# Initialize tokenizer and zero-shot classification pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_ID,
    tokenizer=tokenizer,
    multi_label=True,  # return scores for ALL labels (not just top-1)
    device_map="auto",  # auto-GPU if available
)


def classifier() -> Any:
    """Return the underlying HuggingFace zero-shot classification pipeline."""
    return _classifier


def _softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax activation function to logits.

    Args:
        logits: Input tensor of raw logits.
        dim: The dimension along which to apply softmax. Defaults to last dimension.

    Returns:
        Tensor with softmax probabilities.
    """
    return F.softmax(logits, dim=dim)


def flatten_with_groups(
    hierarchical_labels: dict[str, list[str]],
) -> tuple[list[str], dict[str, str]]:
    """Flatten hierarchical labels to dot-notation and build label->group map.

    Args:
        hierarchical_labels: Dictionary mapping group names to lists of labels.

    Returns:
        A tuple containing:
            - List of flattened labels in dot-notation (e.g., "group.label")
            - Dictionary mapping each flat label to its group name.
    """
    flat: list[str] = []
    label_to_group: dict[str, str] = {}
    for group, labels in hierarchical_labels.items():
        for label in labels:
            flat_label = f"{group}.{label}"
            flat.append(flat_label)
            label_to_group[flat_label] = group
    return flat, label_to_group


def per_group_rescale(
    scores: list[float],
    flat_labels: list[str],
    label_to_group: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Renormalize raw NLI scores per group via softmax.

    NLI scores are already in [0, 1] (sigmoid-normalized). We convert them
    to logits via inverse-sigmoid, then apply softmax within each group to
    produce a proper categorical distribution.

    Args:
        scores: Raw NLI entailment scores (one per flat label).
        flat_labels: List of flattened label strings.
        label_to_group: Dictionary mapping each flat label to its group.

    Returns:
        Dictionary mapping group names to renormalized probability tensors.
    """
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(flat_labels):
        group = label_to_group[label]
        groups.setdefault(group, []).append(i)

    rescaled: dict[str, torch.Tensor] = {}
    for group, indices in groups.items():
        group_scores = torch.tensor([scores[i] for i in indices])
        # Clamp to avoid log(0), convert sigmoid scores to logits, then softmax
        eps = 1e-7
        logits = torch.log(group_scores.clamp(eps, 1 - eps) / (1 - group_scores).clamp(eps, 1 - eps))
        rescaled[group] = _softmax(logits)
    return rescaled


def jev_choice_answer(
    group: str,
    group_probs: torch.Tensor,
    flat_labels: list[str],
    label_to_group: dict[str, str],
) -> dict:
    """Build a TypeSafe Choice-shaped answer for one group.

    Args:
        group: The group name for this choice.
        group_probs: Renormalized probability tensor for this group.
        flat_labels: List of all flattened label strings.
        label_to_group: Dictionary mapping flat labels to their groups.

    Returns:
        Dictionary containing type, choice, confidence, and probabilities.
    """
    flat_labels_for_group = [l for l in flat_labels if label_to_group[l] == group]
    choice_key = int(group_probs.argmax().item())
    choice_label = flat_labels_for_group[choice_key]
    probs = {flat_labels_for_group[i]: float(group_probs[i]) for i in range(len(group_probs))}
    confidence = float(group_probs.max())
    return {
        "type": "choice",
        "choice": choice_label,
        "confidence": confidence,
        "probabilities": probs,
    }


def adapt_choice(
    texts: list[str],
    hierarchical_labels: dict[str, list[str]],
    prompt: str | None = None,
) -> list[dict]:
    """Convert HF zero-shot single-label to per-group renormalized TypeSafe Choice answers.

    All texts share the same label set and are processed in a single pipeline call
    via ``candidate_labels``.

    Args:
        texts: List of input texts to classify.
        hierarchical_labels: Dictionary mapping group names to lists of labels.
        prompt: Optional prefix prompt (e.g. "Given the following statement").

    Returns:
        List of dictionaries containing flat_labels, flat_probabilities, and answers.
    """
    flat_labels, label_to_group = flatten_with_groups(hierarchical_labels)
    answers: list[dict] = []

    results = _classifier(texts, flat_labels, prompt=prompt, multi_label=True)

    for res in results:
        scores = res["scores"]
        flat_probs = _softmax(torch.tensor(scores))
        group_rescales = per_group_rescale(scores, flat_labels, label_to_group)

        group_answers: dict[str, dict] = {}
        for group in hierarchical_labels:
            group_answers[group] = jev_choice_answer(group, group_rescales[group], flat_labels, label_to_group)

        answers.append(
            {
                "flat_labels": flat_labels,
                "flat_probabilities": {flat_labels[i]: float(flat_probs[i]) for i in range(len(flat_labels))},
                "answers": group_answers,
            }
        )

    return answers


def adapt_noul(
    texts: list[str],
    instructions: str | list[str],
    prompt: str | None = None,
) -> list[dict]:
    """Convert HF zero-shot binary classification to TypeSafe Noul answers.

    All texts share the same labels (yes/no) and are processed in a single
    pipeline call.

    Args:
        texts: List of input texts to classify.
        instructions: Single instruction string or list of instructions for each text.
        prompt: Optional prefix prompt.

    Returns:
        List of dictionaries containing type, noul, confidence, and probabilities.
    """
    labels = ["yes", "no"]
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    results = _classifier(texts, labels, multi_label=True)

    answers = []
    for res, instr in zip(results, instructions):
        scores = res["scores"]
        probs = _softmax(torch.tensor(scores))
        yes_prob = float(probs[0])
        noul = 1.0 if yes_prob >= 0.5 else 0.0
        confidence = float(probs.max())
        answers.append(
            {
                "type": "noul",
                "noul": noul,
                "confidence": confidence,
                "probabilities": {
                    "yes": float(probs[0]),
                    "no": float(probs[1]),
                },
            }
        )

    return answers


def _score_from_rank(probs: torch.Tensor) -> tuple[float, float]:
    """Compute a fractional ordinal score from a tensor of class probabilities.

    Computes the **expected value** of the level indices under the
    probability distribution, yielding a soft ordinal score in ``[0, N-1]``.
    A model that is ``"almost sure"`` about level 2 but has meaningful
    probability on level 3 will yield a fractional score between 2 and 3.

    Confidence is the maximum class probability.

    Args:
        probs: 1-D tensor of softmax probabilities for ordered levels.

    Returns:
        A tuple of (fractional_score, confidence) both as Python floats.
    """
    indices = torch.arange(len(probs), dtype=probs.dtype, device=probs.device)
    score = float((indices * probs).sum().item())
    confidence = float(probs.max())
    return score, confidence


def adapt_score(
    texts: list[str],
    instructions: str | list[str],
    criteria: list[str],
    prompt: str | None = None,
) -> list[dict]:
    """Convert HF zero-shot ordinal classification to TypeSafe Score answers.

    All texts share the same labels (criteria) and are processed in a single
    pipeline call.

    The ordinal score is a **fractional** value computed via reciprocal-rank
    weighting: the most probable level (rank 1) gets weight 1, the next
    (rank 2) gets weight 1/2, etc.

    Args:
        texts: List of input texts to classify.
        instructions: Single instruction string or list of instructions for each text.
        criteria: List of ordered criteria/levels for scoring.
        prompt: Optional prefix prompt.

    Returns:
        List of dictionaries containing type, score, confidence, legend, and probabilities.
    """
    labels = criteria
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    results = _classifier(texts, labels, multi_label=True)
    legend = {str(i): level for i, level in enumerate(criteria)}

    answers = []
    for res, instr in zip(results, instructions):
        scores = res["scores"]
        probs = _softmax(torch.tensor(scores))
        score, confidence = _score_from_rank(probs)
        answers.append(
            {
                "type": "score",
                "score": score,
                "confidence": confidence,
                "legend": legend,
                "probabilities": {str(i): float(probs[i]) for i in range(len(probs))},
            }
        )

    return answers


def jev_api(
    texts: list[str],
    questions: dict[str, dict],
    prompt: str | None = None,
) -> list[dict]:
    """Unified entry point matching the Typesafe API request/response contract.

    All texts and ALL questions are batched into a SINGLE pipeline call per text
    via ``candidate_labels`` with hierarchical labels.

    Args:
        texts: List of input texts to classify.
        questions: Dictionary of named questions with type, instructions, and criteria.
            Example structure:
            {
                "department": {
                    "type": "choice",
                    "instructions": str,
                    "criteria": {"key": "desc", ...}  # hierarchical labels
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": str
                },
                "frustration": {
                    "type": "score",
                    "instructions": str,
                    "criteria": ["level 0", "level 1", ...]
                }
            }
        prompt: Optional prefix prompt prepended to each text.

    Returns:
        List of answer dictionaries, one per input text, with structure:
        [
            {
                "model": "jev-hf",
                "answers": {
                    "department": {"type": "choice", "choice": "...", ...},
                    "is_urgent": {"type": "noul", "noul": 1.0, ...},
                    "frustration": {"type": "score", "score": 1.0, ...}
                }
            },
            ...
        ]
    """
    # Build unified hierarchical labels and question metadata
    hierarchical_labels: dict[str, list[str]] = {}
    question_meta: dict[str, dict] = {}

    for name, question in questions.items():
        q_type = question["type"]
        criteria = question.get("criteria")
        instructions = question["instructions"]

        if q_type == "choice":
            if isinstance(criteria, dict):
                hierarchical_labels[name] = list(criteria.keys())
            else:
                assert isinstance(criteria, list)
                hierarchical_labels[name] = criteria
        elif q_type == "noul":
            hierarchical_labels[name] = ["yes", "no"]
        elif q_type == "score":
            if isinstance(criteria, dict):
                hierarchical_labels[name] = list(criteria.keys())
            else:
                assert isinstance(criteria, list)
                hierarchical_labels[name] = criteria
        else:
            raise ValueError(f"Unknown question type: {q_type}")

        question_meta[name] = {
            "type": q_type,
            "instructions": instructions,
        }

    # Flatten hierarchical labels
    flat_labels, label_to_group = flatten_with_groups(hierarchical_labels)

    # Build unified prompt that includes all instructions
    unified_instructions = " | ".join(f"{name}: {meta['instructions']}" for name, meta in question_meta.items())

    # Single pipeline call per text with ALL candidate labels
    results = _classifier(texts, flat_labels, prompt=prompt, multi_label=True)

    # Post-process results
    per_text_answers: list[dict[str, dict]] = []

    for res in results:
        scores = res["scores"]
        group_rescales = per_group_rescale(scores, flat_labels, label_to_group)

        text_answers: dict[str, dict] = {}

        for name, meta in question_meta.items():
            q_type = meta["type"]
            group_probs = group_rescales[name]

            if q_type == "choice":
                flat_labels_for_group = [l for l in flat_labels if label_to_group[l] == name]
                choice_key = int(group_probs.argmax().item())
                choice_label = flat_labels_for_group[choice_key].split(".")[-1]
                probs = {
                    flat_labels_for_group[i].split(".")[-1]: float(group_probs[i]) for i in range(len(group_probs))
                }
                confidence = float(group_probs.max())
                text_answers[name] = {
                    "type": "choice",
                    "choice": choice_label,
                    "confidence": confidence,
                    "probabilities": probs,
                }

            elif q_type == "noul":
                yes_prob = float(group_probs[0])
                noul = 1.0 if yes_prob >= 0.5 else 0.0
                confidence = float(group_probs.max())
                text_answers[name] = {
                    "type": "noul",
                    "noul": noul,
                    "confidence": confidence,
                    "probabilities": {
                        "yes": float(group_probs[0]),
                        "no": float(group_probs[1]),
                    },
                }

            elif q_type == "score":
                score, confidence = _score_from_rank(group_probs)
                probs = {str(i): float(group_probs[i]) for i in range(len(group_probs))}
                text_answers[name] = {
                    "type": "score",
                    "score": score,
                    "confidence": confidence,
                    "legend": {str(i): hierarchical_labels[name][i] for i in range(len(hierarchical_labels[name]))},
                    "probabilities": probs,
                }

        per_text_answers.append(text_answers)

    return [{"model": "jev-hf", "answers": answers} for answers in per_text_answers]


def main() -> None:
    texts = [
        "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
    ]

    print("\n--- Unified JEV API (HuggingFace zero-shot) ---")
    api_results = jev_api(
        texts=texts,
        questions={
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this",
                "criteria": {
                    "billing": "Payment or subscription issues",
                    "technical": "Bugs or integration problems",
                    "sales": "Pricing or account questions",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated the customer appears",
                "criteria": [
                    "Calm, just stating facts",
                    "Frustrated but civil",
                    "Very angry, strong language",
                ],
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "The message conveys urgency or time-sensitivity",
            },
        },
    )

    for text_idx, result in enumerate(api_results):
        print(f"\n  --- Text {text_idx + 1} ---")
        for name, answer in result["answers"].items():
            if answer["type"] == "choice":
                print(f"  [{name}] choice={answer['choice']}, confidence={answer['confidence']:.4f}")
            elif answer["type"] == "noul":
                print(f"  [{name}] noul={answer['noul']}, confidence={answer['confidence']:.4f}")
            elif answer["type"] == "score":
                print(f"  [{name}] score={answer['score']:.4f}, confidence={answer['confidence']:.4f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
