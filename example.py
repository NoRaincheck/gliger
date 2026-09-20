"""TypeSafe Choice/Noul/Score primitives emulation using GLiClass logits.

This module provides adapters that emulate TypeSafe API primitives (Choice, Noul, Score)
using GLiClass model logits. It follows the Typesafe API contract:

- Choice  -> multi-class probabilities (per-group renormalized softmax)
- Noul    -> binary yes/no probability (0=no, 1=yes)
- Score   -> ordered categorical (0..N with legend + probabilities)

The module uses batch GLiClass inference via get_embeddings for all three primitives,
enabling efficient processing of multiple texts and questions in a single forward pass.

See: https://docs.typesafe.ai/introduction/quickstart
"""

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

# Model configuration
MODEL_ID = "knowledgator/gliclass-large-v1.0"

# Initialize model and pipeline
model = GLiClassModel.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type="single-label"
)


def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax activation function to logits.

    Args:
        logits: Input tensor of raw logits.
        dim: The dimension along which to apply softmax. Defaults to last dimension.

    Returns:
        Tensor with softmax probabilities.
    """
    return F.softmax(logits, dim=dim)


def get_logits(
    text: str | list[str],
    labels: list[str],
    prompt: str | list[str] | None = None,
    examples: list[dict] | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """Get raw logits via get_embeddings. Supports single or batch text.

    Args:
        text: Single text string or list of texts to classify.
        labels: List of label strings for classification.
        prompt: Optional prompt or list of prompts to guide classification.
        examples: Optional list of example dictionaries for few-shot learning.

    Returns:
        Single tensor of logits for one text, or list of tensors for batch input.
    """
    emb = pipeline.get_embeddings(text, labels, prompt=prompt, examples=examples)
    if isinstance(text, str):
        return torch.tensor(emb[0]["logits"])
    return [torch.tensor(e["logits"]) for e in emb]


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
    flat = pipeline.flatten_labels(hierarchical_labels)
    label_to_group: dict[str, str] = {}
    for group, labels in hierarchical_labels.items():
        for label in labels:
            label_to_group[f"{group}.{label}"] = group
    return flat, label_to_group


def per_group_rescale(
    logits: torch.Tensor,
    flat_labels: list[str],
    label_to_group: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Renormalize flat probabilities per group via softmax on the logits.

    Args:
        logits: Raw logits tensor from the model.
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
        group_logits = logits[indices]
        rescaled[group] = softmax(group_logits)
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
    probs = {
        flat_labels_for_group[i]: float(group_probs[i]) for i in range(len(group_probs))
    }
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
    examples: list[dict] | None = None,
) -> list[dict]:
    """Convert GLiClass batched single-label to per-group renormalized TypeSafe Choice answers.

    All texts share the same label set and are processed in a single GLiClass
    forward pass via the batch ``get_embeddings`` interface.

    Args:
        texts: List of input texts to classify.
        hierarchical_labels: Dictionary mapping group names to lists of labels.
        examples: Optional list of example dictionaries for few-shot learning.

    Returns:
        List of dictionaries containing flat_labels, flat_probabilities, and answers.
    """
    flat_labels, label_to_group = flatten_with_groups(hierarchical_labels)
    logits_list = get_logits(texts, flat_labels, examples=examples)

    answers: list[dict] = []
    for logits in logits_list:
        flat_probs = softmax(logits)
        group_rescales = per_group_rescale(logits, flat_labels, label_to_group)

        group_answers: dict[str, dict] = {}
        for group in hierarchical_labels:
            group_answers[group] = jev_choice_answer(
                group, group_rescales[group], flat_labels, label_to_group
            )

        answers.append(
            {
                "flat_labels": flat_labels,
                "flat_probabilities": {
                    flat_labels[i]: float(flat_probs[i])
                    for i in range(len(flat_labels))
                },
                "answers": group_answers,
            }
        )

    return answers


def adapt_noul(
    texts: list[str],
    instructions: str | list[str],
    examples: list[dict] | None = None,
) -> list[dict]:
    """Convert GLiClass batched binary classification to TypeSafe Noul answers.

    All texts share the same labels (yes/no) and are processed in a single
    GLiClass forward pass via the batch ``get_embeddings`` interface.

    Args:
        texts: List of input texts to classify.
        instructions: Single instruction string or list of instructions for each text.
        examples: Optional list of example dictionaries for few-shot learning.

    Returns:
        List of dictionaries containing type, noul, confidence, and probabilities.
    """
    labels = ["yes", "no"]
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    logits_list = get_logits(texts, labels, prompt=instructions, examples=examples)

    answers = []
    for logits in logits_list:
        probs = softmax(logits)
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


def _score_from_rank(
    probs: torch.Tensor,
) -> tuple[float, float]:
    """Compute a fractional ordinal score from a tensor of class probabilities.

    Ranks classes by probability (descending), then computes a weighted
    score where each level index is weighted by its rank-adjusted
    probability contribution. Confidence is the maximum class probability.

    Args:
        probs: 1-D tensor of softmax probabilities for ordered levels.

    Returns:
        A tuple of (fractional_score, confidence) both as Python floats.
    """
    ranked = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)
    weighted_sum = 0.0
    score = 0.0
    for level_idx, prob in ranked:
        weight = level_idx * prob
        weighted_sum += weight
        current_weight_prop = weight / weighted_sum

        score = level_idx * (current_weight_prop) + score * (1 - current_weight_prop)

    return float(score), float(probs.max())


def adapt_score(
    texts: list[str],
    instructions: str | list[str],
    criteria: list[str],
    examples: list[dict] | None = None,
) -> list[dict]:
    """Convert GLiClass batched ordinal classification to TypeSafe Score answers.

    All texts share the same labels (criteria) and are processed in a single
    GLiClass forward pass via the batch ``get_embeddings`` interface.

    The ordinal score is a **fractional** value computed via reciprocal-rank
    weighting: the most probable level (rank 1) gets weight 1, the next
    (rank 2) gets weight 1/2, etc., each multiplied by its probability.

    Args:
        texts: List of input texts to classify.
        instructions: Single instruction string or list of instructions for each text.
        criteria: List of ordered criteria/levels for scoring.
        examples: Optional list of example dictionaries for few-shot learning.

    Returns:
        List of dictionaries containing type, score, confidence, legend, and probabilities.
    """
    labels = criteria  # e.g. ["Calm", "Frustrated", "Very angry"]
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    logits_list = get_logits(texts, labels, prompt=instructions, examples=examples)
    legend = {str(i): level for i, level in enumerate(criteria)}

    answers = []
    for logits in logits_list:
        probs = softmax(logits)
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
    examples: list[dict] | None = None,
) -> list[dict]:
    """Unified entry point matching the Typesafe API request/response contract.

    All texts and ALL questions are batched into a SINGLE GLiClass forward pass
    via the batch ``get_embeddings`` interface with hierarchical labels.

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
        examples: Optional list of example dictionaries for few-shot learning.

    Returns:
        List of answer dictionaries, one per input text, with structure:
        [
            {
                "model": "jev-local",
                "answers": {
                    "department": {"type": "choice", "choice": "...", ...},
                    "is_urgent": {"type": "noul", "noul": 1.0, ...},
                    "frustration": {"type": "score", "score": 1.0, ...}
                }
            },
            ...
        ]
    """
    # Build unified hierarchical labels: {question_name: [answer_options, ...]}
    hierarchical_labels: dict[str, list[str]] = {}
    question_meta: dict[str, dict] = {}  # store type + instructions per question

    for name, question in questions.items():
        q_type = question["type"]
        criteria = question.get("criteria")
        instructions = question["instructions"]

        if q_type == "choice":
            # criteria is dict {label: desc} or hierarchical
            if isinstance(criteria, dict):
                hierarchical_labels[name] = list(criteria.keys())
            else:
                hierarchical_labels[name] = criteria
        elif q_type == "noul":
            hierarchical_labels[name] = ["yes", "no"]
        elif q_type == "score":
            # criteria is list or dict
            if isinstance(criteria, dict):
                hierarchical_labels[name] = list(criteria.keys())
            else:
                hierarchical_labels[name] = criteria
        else:
            raise ValueError(f"Unknown question type: {q_type}")

        question_meta[name] = {
            "type": q_type,
            "instructions": instructions,
        }

    # Single batched GLiClass call for ALL texts and ALL questions
    flat_labels, label_to_group = flatten_with_groups(hierarchical_labels)

    # Build unified prompt that includes all instructions
    # Format: "Q1: <instr1> | Q2: <instr2> | ..."
    unified_instructions = " | ".join(
        f"{name}: {meta['instructions']}" for name, meta in question_meta.items()
    )
    prompts = [unified_instructions] * len(texts)

    logits_list = get_logits(texts, flat_labels, prompt=prompts, examples=examples)

    # Post-process results
    per_text_answers: list[dict[str, dict]] = []

    for logits in logits_list:
        flat_probs = softmax(logits)
        group_rescales = per_group_rescale(logits, flat_labels, label_to_group)

        text_answers: dict[str, dict] = {}

        for name, meta in question_meta.items():
            q_type = meta["type"]
            group_probs = group_rescales[name]

            if q_type == "choice":
                flat_labels_for_group = [
                    l for l in flat_labels if label_to_group[l] == name
                ]
                choice_key = int(group_probs.argmax().item())
                choice_label = flat_labels_for_group[choice_key].split(".")[
                    -1
                ]  # strip group prefix
                probs = {
                    flat_labels_for_group[i].split(".")[-1]: float(group_probs[i])
                    for i in range(len(group_probs))
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
                    "legend": {
                        str(i): hierarchical_labels[name][i]
                        for i in range(len(hierarchical_labels[name]))
                    },
                    "probabilities": probs,
                }

        per_text_answers.append(text_answers)

    return [{"model": "jev-local", "answers": answers} for answers in per_text_answers]


def main() -> None:
    texts = [
        "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
    ]

    print("\n--- Unified Jev API (batched inference) ---")
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
                print(
                    f"  [{name}] choice={answer['choice']}, confidence={answer['confidence']:.4f}"
                )
            elif answer["type"] == "noul":
                print(
                    f"  [{name}] noul={answer['noul']}, confidence={answer['confidence']:.4f}"
                )
            elif answer["type"] == "score":
                print(
                    f"  [{name}] score={answer['score']:.4f}, confidence={answer['confidence']:.4f}"
                )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
