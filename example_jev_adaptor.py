"""
example_jev_adaptor.py — emulate TypeSafe Choice/Noul/Score primitives using GLiClass logits.

Follows the Typesafe API contract (https://docs.typesafe.ai/introduction/quickstart):

  - Choice  → multi-class probabilities  (per-group renormalized softmax)
  - Noul    → binary yes/no probability  (0=no, 1=yes)
  - Score   → ordered categorical        (0..N with legend + probabilities)

Uses batch GLiClass inference via get_embeddings for all three primitives.
"""

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

# ── model setup ───────────────────────────────────────────────────────────────
model_id = "knowledgator/gliclass-base-v1.0"  # Use base variant for memory-constrained environments

model = GLiClassModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type="single-label"
)


# ── helpers ───────────────────────────────────────────────────────────────────
def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(logits, dim=dim)


def get_logits(
    text: str | list[str],
    labels: list[str],
    prompt: str | list[str] | None = None,
    examples: list[dict] | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """Get raw logits via get_embeddings. Supports single or batch text."""
    emb = pipeline.get_embeddings(text, labels, prompt=prompt, examples=examples)
    if isinstance(text, str):
        return torch.tensor(emb[0]["logits"])
    return [torch.tensor(e["logits"]) for e in emb]


def flatten_with_groups(
    hierarchical_labels: dict[str, list[str]],
) -> tuple[list[str], dict[str, str]]:
    """Flatten hierarchical labels to dot-notation and build label→group map."""
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
    """Renormalize flat probabilities per group via softmax on the logits."""
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
    """Build a TypeSafe Choice-shaped answer for one group."""
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


# ── Choice primitive (batched) ────────────────────────────────────────────────
def adapt_choice(
    texts: list[str],
    hierarchical_labels: dict[str, list[str]],
    examples: list[dict] | None = None,
) -> list[dict]:
    """GLiClass batched single-label → per-group renormalized TypeSafe Choice answers.

    All texts share the same label set and are processed in a single GLiClass
    forward pass via the batch ``get_embeddings`` interface.
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


# ── Noul primitive (binary yes/no, batched) ──────────────────────────────────
def adapt_noul(
    texts: list[str],
    instructions: str | list[str],
    examples: list[dict] | None = None,
) -> list[dict]:
    """GLiClass batched binary classification → list of TypeSafe Noul answers.

    All texts share the same labels (yes/no) and are processed in a single
    GLiClass forward pass via the batch ``get_embeddings`` interface.
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


# ── Score primitive (ordered categorical, batched) ───────────────────────────
def adapt_score(
    texts: list[str],
    instructions: str | list[str],
    criteria: list[str],
    examples: list[dict] | None = None,
) -> list[dict]:
    """GLiClass batched ordinal classification → list of TypeSafe Score answers.

    All texts share the same labels (criteria) and are processed in a single
    GLiClass forward pass via the batch ``get_embeddings`` interface.
    """
    labels = criteria  # e.g. ["Calm", "Frustrated", "Very angry"]
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    logits_list = get_logits(texts, labels, prompt=instructions, examples=examples)
    legend = {str(i): level for i, level in enumerate(criteria)}

    answers = []
    for logits in logits_list:
        probs = softmax(logits)
        score_idx = int(probs.argmax().item())
        answers.append(
            {
                "type": "score",
                "score": float(score_idx),
                "confidence": float(probs.max()),
                "legend": legend,
                "probabilities": {str(i): float(probs[i]) for i in range(len(probs))},
            }
        )

    return answers


# ── Unified Jev API (fully batched in one forward pass) ──────────────────────
def jev_api(
    texts: list[str],
    questions: dict[str, dict],
    examples: list[dict] | None = None,
) -> list[dict]:
    """Unified entry point matching the Typesafe API request/response contract.

    All texts and ALL questions are batched into a SINGLE GLiClass forward pass
    via the batch ``get_embeddings`` interface with hierarchical labels.

    Request shape:
        {
            "texts": [str, ...],                   # one or more input texts
            "questions": {                          # named questions
                "department": {
                    "type": "choice",
                    "instructions": str,
                    "criteria": {"key": "desc", ...}   # hierarchical labels
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
        }

    Response shape (one answer dict per input text):
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
                flat_labels_for_group = [l for l in flat_labels if label_to_group[l] == name]
                choice_key = int(group_probs.argmax().item())
                choice_label = flat_labels_for_group[choice_key].split(".")[-1]  # strip group prefix
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
                score_idx = int(group_probs.argmax().item())
                probs = {str(i): float(group_probs[i]) for i in range(len(group_probs))}
                confidence = float(group_probs.max())
                text_answers[name] = {
                    "type": "score",
                    "score": float(score_idx),
                    "confidence": confidence,
                    "legend": {str(i): hierarchical_labels[name][i] for i in range(len(hierarchical_labels[name]))},
                    "probabilities": probs,
                }

        per_text_answers.append(text_answers)

    return [{"model": "jev-local", "answers": answers} for answers in per_text_answers]


# ── run ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Batched texts (one GLiClass forward pass per question type) ─────────
    texts = [
        "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
    ]

    # ── Unified Jev API (batched — all texts per question type) ─────────────
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
                    f"  [{name}] score={answer['score']}, confidence={answer['confidence']:.4f}"
                )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
