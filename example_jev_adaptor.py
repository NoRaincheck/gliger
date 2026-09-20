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
model_id = "knowledgator/gliclass-large-v1.0"

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


# ── Choice primitive (existing) ───────────────────────────────────────────────
def adapt_choice(
    text: str,
    hierarchical_labels: dict[str, list[str]],
    examples: list[dict] | None = None,
) -> dict:
    """GLiClass single-label → per-group renormalized TypeSafe Choice answers."""
    flat_labels, label_to_group = flatten_with_groups(hierarchical_labels)
    logits = get_logits(text, flat_labels, examples=examples)

    # Global flat distribution
    flat_probs = softmax(logits)

    # Per-group renormalized distributions
    group_rescales = per_group_rescale(logits, flat_labels, label_to_group)

    # TypeSafe Choice-shaped answers per group
    answers: dict[str, dict] = {}
    for group in hierarchical_labels:
        answers[group] = jev_choice_answer(
            group, group_rescales[group], flat_labels, label_to_group
        )

    return {
        "flat_labels": flat_labels,
        "flat_probabilities": {
            flat_labels[i]: float(flat_probs[i]) for i in range(len(flat_labels))
        },
        "answers": answers,
    }


# ── Noul primitive (binary yes/no) ───────────────────────────────────────────
def adapt_noul(
    text: str,
    instructions: str,
    examples: list[dict] | None = None,
) -> dict:
    """GLiClass binary classification → TypeSafe Noul answer.

    Uses batch inference with a single text → single yes/no result.
    """
    labels = ["yes", "no"]
    prompt = instructions  # e.g. "Does this message express urgency?"

    logits = get_logits(text, labels, prompt=prompt, examples=examples)
    probs = softmax(logits)

    yes_prob = float(probs[0])  # index 0 = "yes"
    noul = 1.0 if yes_prob >= 0.5 else 0.0
    confidence = float(probs.max())

    return {
        "type": "noul",
        "noul": noul,
        "confidence": confidence,
        "probabilities": {
            "yes": float(probs[0]),
            "no": float(probs[1]),
        },
    }


def adapt_noul_batch(
    texts: list[str],
    instructions: str,
    examples: list[dict] | None = None,
) -> dict:
    """Batch GLiClass inference for multiple texts → list of Noul answers."""
    labels = ["yes", "no"]
    prompts = [instructions] * len(texts)

    logits_list = get_logits(texts, labels, prompt=prompts, examples=examples)

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

    return {"answers": answers}


# ── Score primitive (ordered categorical) ─────────────────────────────────────
def adapt_score(
    text: str,
    instructions: str,
    criteria: list[str],
    examples: list[dict] | None = None,
) -> dict:
    """GLiClass ordinal classification → TypeSafe Score answer.

    Each criterion level gets a numeric index 0..N-1.
    Returns score (index of max prob), legend, and per-level probabilities.
    """
    labels = criteria  # e.g. ["Calm", "Frustrated", "Very angry"]
    prompt = instructions

    logits = get_logits(text, labels, prompt=prompt, examples=examples)
    probs = softmax(logits)

    score_idx = int(probs.argmax().item())
    score = float(score_idx)
    confidence = float(probs.max())

    # legend: {"0": "Calm, just stating facts", "1": "Frustrated but civil", ...}
    legend = {str(i): level for i, level in enumerate(criteria)}
    probabilities = {str(i): float(probs[i]) for i in range(len(probs))}

    return {
        "type": "score",
        "score": score,
        "confidence": confidence,
        "legend": legend,
        "probabilities": probabilities,
    }


def adapt_score_batch(
    texts: list[str],
    instructions: str,
    criteria: list[str],
    examples: list[dict] | None = None,
) -> dict:
    """Batch GLiClass inference for multiple texts → list of Score answers."""
    labels = criteria
    prompts = [instructions] * len(texts)

    logits_list = get_logits(texts, labels, prompt=prompts, examples=examples)
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

    return {"answers": answers}


# ── Unified Jev API (TypeSafe contract) ──────────────────────────────────────
def jev_api(
    state: str,
    questions: dict[str, dict],
    examples: list[dict] | None = None,
) -> dict:
    """Unified entry point matching the Typesafe API request/response contract.

    Request shape:
        {
            "state": str,                          # input text
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

    Response shape:
        {
            "model": "jev-local",
            "answers": {
                "department": {"type": "choice", "choice": "...", ...},
                "is_urgent": {"type": "noul", "noul": 1.0, ...},
                "frustration": {"type": "score", "score": 1.0, ...}
            }
        }
    """
    answers: dict[str, dict] = {}

    for name, question in questions.items():
        q_type = question["type"]
        instructions = question["instructions"]
        criteria = question.get("criteria")

        if q_type == "choice":
            # Convert flat dict {label: desc} to hierarchical {group: [labels]}
            if isinstance(criteria, dict):
                choice_labels = list(criteria.keys())
                hierarchical = {name: choice_labels}
            else:
                hierarchical = criteria
            result = adapt_choice(state, hierarchical, examples)
            answers[name] = result["answers"].get(name, {})
            # Flatten the dot-notation label names back to bare labels
            if answers[name]:
                flat_probs = {
                    k.split(".")[-1]: v
                    for k, v in answers[name].get("probabilities", {}).items()
                }
                answers[name]["probabilities"] = flat_probs
                answers[name]["choice"] = answers[name]["choice"].split(".")[-1]
        elif q_type == "noul":
            answers[name] = adapt_noul(state, instructions, examples)
        elif q_type == "score":
            answers[name] = adapt_score(state, instructions, criteria, examples)
        else:
            raise ValueError(f"Unknown question type: {q_type}")

    return {
        "model": "jev-local",
        "answers": answers,
    }


# ── run ───────────────────────────────────────────────────────────────────────
def main() -> None:
    state = "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP."

    # ── Unified Jev API ─────────────────────────────────────────────────────
    print("\n--- Unified Jev API (all three primitives) ---")
    api_result = jev_api(
        state=state,
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

    for name, answer in api_result["answers"].items():
        print(f"\n  [{name}]")
        if answer["type"] == "choice":
            print(f"  choice:     {answer['choice']}")
            print(f"  confidence: {answer['confidence']:.4f}")
        elif answer["type"] == "noul":
            print(f"  noul:       {answer['noul']}")
            print(f"  confidence: {answer['confidence']:.4f}")
        elif answer["type"] == "score":
            print(f"  score:      {answer['score']}")
            print(f"  confidence: {answer['confidence']:.4f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
