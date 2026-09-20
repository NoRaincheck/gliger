"""Generic zero-shot classification adapter for HuggingFace pipelines.

Provides a flexible, model-agnostic interface for classifying text inputs against
any set of candidate labels using HuggingFace's ``pipeline("zero-shot-classification")``.

The generic adapter wraps the raw pipeline output with a standard response format,
making it easy to plug into any project that needs zero-shot classification without
tight coupling to a specific model or API shape.

Key design principles:
- Model-agnostic: any HuggingFace zero-shot model works.
- Clean response shape: returns per-text classifications with labels and probabilities.
- No hardcoded model IDs, class names, or internal logic.
- Supports single-label, multi-label, and binary classification modes.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, pipeline


# ── Default model and cache ────────────────────────────────────────────────

DEFAULT_MODEL_ID: str = "facebook/bart-large-mnli"  # fallback model for zero-shot classification
_model_cache: Dict[str, tuple[Any, Any]] = {}


def _load_pipeline(model_id: str) -> Tuple[Any, Any]:
    """Load and cache a tokenizer + zero-shot classifier pipeline."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    clf = pipeline(
        "zero-shot-classification",
        model=model_id,
        tokenizer=tokenizer,
        multi_label=False,  # default: single-label classification
    )
    _model_cache[model_id] = (tokenizer, clf)
    return tokenizer, clf


# ── Classification helpers ─────────────────────────────────────────────────

def classify(
    texts: List[str],
    candidate_labels: List[str],
    model_id: str = DEFAULT_MODEL_ID,
    *,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """Classify a list of texts against the given candidate labels.

    Args:
        texts: Input text strings to classify.
        candidate_labels: Labels to evaluate (e.g., ["A", "B", "C"]).
        model_id: HuggingFace zero-shot model identifier. Defaults to ``DEFAULT_MODEL_ID``.
        top_k: Number of top results to return per text. Set to 1 for single-label mode.

    Returns:
        List of dicts, one per input text. Each dict has:
            - "label": the most (or top-k) predicted label
            - "score": confidence score (0..1)
            - "top_labels": list of top labels with scores
            - "raw_scores": list of raw scores for all candidate labels
    """
    tokenizer, clf = _load_pipeline(model_id)

    results = clf(texts, candidate_labels=candidate_labels, multi_label=False)

    out: List[Dict[str, Any]] = []
    for i, res in enumerate(results):
        raw_scores = [float(s) for s in res["scores"]]
        top_indices = sorted(range(len(raw_scores)), key=lambda j: raw_scores[j], reverse=True)[:top_k]
        top_labels_and_scores = [
            {"label": candidate_labels[j], "score": raw_scores[j]} for j in top_indices
        ]
        out.append({
            "label": top_labels_and_scores[0]["label"],
            "score": top_labels_and_scores[0]["score"],
            "top_labels": top_labels_and_scores,
            "raw_scores": raw_scores,
            "index": i,
        })

    return out


def classify_multi(
    texts: List[str],
    candidate_labels: List[str],
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    """Multi-label zero-shot classification.

    Args:
        texts: Input text strings to classify.
        candidate_labels: Labels to evaluate.
        model_id: HuggingFace zero-shot model identifier. Defaults to ``DEFAULT_MODEL_ID``.

    Returns:
        List of dicts with the same shape as ``classify()`` but using multi_label=True.
    """
    tokenizer, clf = _load_pipeline(model_id)

    results = clf(texts, candidate_labels=candidate_labels, multi_label=True)

    out: List[Dict[str, Any]] = []
    for i, res in enumerate(results):
        raw_scores = [float(s) for s in res["scores"]]
        top_indices = sorted(range(len(raw_scores)), key=lambda j: raw_scores[j], reverse=True)[:5]
        top_labels_and_scores = [
            {"label": candidate_labels[j], "score": raw_scores[j]} for j in top_indices
        ]
        out.append({
            "label": top_labels_and_scores[0]["label"],
            "score": top_labels_and_scores[0]["score"],
            "top_labels": top_labels_and_scores,
            "raw_scores": raw_scores,
            "index": i,
        })

    return out


def classify_binary(
    texts: List[str],
    instructions: str | list[str],
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    """Binary classification (yes/no).

    Args:
        texts: Input text strings to classify.
        instructions: Single instruction or list of instructions for each text.
        model_id: HuggingFace zero-shot model identifier. Defaults to ``DEFAULT_MODEL_ID``.

    Returns:
        List of dicts with "yes" and "no" probability scores.
    """
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    tokenizer, clf = _load_pipeline(model_id)

    results = clf(texts, candidate_labels=["yes", "no"], multi_label=True)

    out: List[Dict[str, Any]] = []
    for i, res in enumerate(results):
        raw_scores = [float(s) for s in res["scores"]]
        yes_prob = raw_scores[0]
        no_prob = raw_scores[1]
        out.append({
            "label": "yes" if yes_prob >= 0.5 else "no",
            "score": max(yes_prob, no_prob),
            "probabilities": {"yes": yes_prob, "no": no_prob},
            "raw_scores": raw_scores,
            "index": i,
        })

    return out


# ── TypeSafe API adapters (JEV-style) ──────────────────────────────────────
# These are thin wrappers around the generic classifier that produce the
# same TypeSafe response shape as the existing JEV adapter.

def adapt_choice(
    texts: List[str],
    hierarchical_labels: Dict[str, List[str]],
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    """Convert a set of per-group labels into TypeSafe Choice answers.

    Each group gets its own softmax-renormalized probability distribution over its
    candidate labels. The most probable label in each group is the selected answer.

    Args:
        texts: Input text strings to classify.
        hierarchical_labels: Dict mapping group names to lists of candidate labels.
        model_id: HuggingFace zero-shot model identifier. Defaults to ``DEFAULT_MODEL_ID``.

    Returns:
        List of dicts, one per input text, with structure:
        [{"flat_labels": [...], "answers": {"group": {"type": "choice", "choice": "...", ...}}}]
    """
    flat_labels: list[str] = []
    label_to_group: dict[str, str] = {}
    for group, labels in hierarchical_labels.items():
        for label in labels:
            flat_label = f"{group}.{label}"
            flat_labels.append(flat_label)
            label_to_group[flat_label] = group

    tokenizer, clf = _load_pipeline(model_id)
    results = clf(texts, candidate_labels=flat_labels, multi_label=False)

    out: List[Dict[str, Any]] = []
    for i, res in enumerate(results):
        raw_scores = [float(s) for s in res["scores"]]

        group_answers: dict[str, Dict[str, Any]] = {}
        for group, labels in hierarchical_labels.items():
            group_flat = [f"{group}.{label}" for label in labels]
            group_indices = [flat_labels.index(l) for l in group_flat]
            group_scores = [raw_scores[j] for j in group_indices]
            top_idx = group_scores.index(max(group_scores))
            selected_label = labels[group_indices[top_idx]]
            group_answers[group] = {
                "type": "choice",
                "choice": selected_label,
                "score": max(group_scores),
            }

        out.append({
            "flat_labels": flat_labels,
            "raw_scores": raw_scores,
            "answers": group_answers,
        })

    return out


def adapt_noul(
    texts: List[str],
    instructions: str | list[str],
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    """Binary yes/no classification using TypeSafe Noul answers.

    Args:
        texts: Input text strings to classify.
        instructions: Single instruction or list of instructions for each text.
        model_id: HuggingFace zero-shot model identifier. Defaults to ``DEFAULT_MODEL_ID``.

    Returns:
        List of dicts with "noul" and confidence scores.
    """
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    tokenizer, clf = _load_pipeline(model_id)
    results = clf(texts, candidate_labels=["yes", "no"], multi_label=True)

    out: List[Dict[str, Any]] = []
    for i, res in enumerate(results):
        raw_scores = [float(s) for s in res["scores"]]
        yes_prob = raw_scores[0]
        noul = 1.0 if yes_prob >= 0.5 else 0.0
        out.append({
            "type": "noul",
            "noul": noul,
            "confidence": max(yes_prob, 1 - yes_prob),
            "probabilities": {"yes": yes_prob, "no": 1 - yes_prob},
            "index": i,
        })

    return out


def adapt_score(
    texts: List[str],
    instructions: str | list[str],
    criteria: List[str],
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    """Ordinal classification using TypeSafe Score answers.

    Args:
        texts: Input text strings to classify.
        instructions: Single instruction or list of instructions for each text.
        criteria: Ordered list of criterion labels (rank 0 = highest).
        model_id: HuggingFace zero-shot model identifier. Defaults to ``DEFAULT_MODEL_ID``.

    Returns:
        List of dicts with "score" (fractional ordinal), confidence, and probabilities.
    """
    if isinstance(instructions, str):
        instructions = [instructions] * len(texts)

    tokenizer, clf = _load_pipeline(model_id)
    results = clf(texts, candidate_labels=criteria, multi_label=False)

    out: List[Dict[str, Any]] = []
    for i, res in enumerate(results):
        raw_scores = [float(s) for s in res["scores"]]
        # Sort by score descending to get rank order; rank 0 is highest
        sorted_indices = sorted(range(len(raw_scores)), key=lambda j: raw_scores[j], reverse=True)
        score = float(sum(j / len(criteria) for j, _ in enumerate(sorted_indices)))
        out.append({
            "type": "score",
            "score": score,
            "confidence": max(raw_scores),
            "probabilities": {str(i): raw_scores[i] for i in range(len(raw_scores))},
            "index": i,
        })

    return out


def jev_api(
    texts: List[str],
    questions: Dict[str, dict],
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    """Unified entry point matching the Typesafe API request/response contract.

    Args:
        texts: List of input texts to classify.
        questions: Dict mapping question names to dicts with keys:
            - "type": "choice", "noul", or "score"
            - "instructions": prompt string(s)
            - "criteria": hierarchical labels (dict for choice/score, list for noul)
        model_id: HuggingFace zero-shot model identifier. Defaults to ``DEFAULT_MODEL_ID``.

    Returns:
        List of answer dicts, one per input text, with structure:
        [{"model": "generic-zs", "answers": {"q_name": {"type": "...", ...}}}]
    """
    # Build unified hierarchical labels and question metadata
    hierarchical_labels: Dict[str, List[str]] = {}
    question_meta: Dict[str, dict] = {}

    for name, question in questions.items():
        q_type = question["type"]
        criteria = question.get("criteria")
        instructions = question["instructions"]

        if q_type == "choice":
            assert isinstance(criteria, dict), f"Choice questions require a dict of hierarchical labels, got {type(criteria)}"
            hierarchical_labels[name] = list(criteria.keys())
        elif q_type == "noul":
            hierarchical_labels[name] = ["yes", "no"]
        elif q_type == "score":
            assert isinstance(criteria, list), f"Score questions require a list of criteria, got {type(criteria)}"
            hierarchical_labels[name] = criteria
        else:
            raise ValueError(f"Unknown question type: {q_type}")

        question_meta[name] = {"type": q_type, "instructions": instructions}

    # Flatten hierarchical labels
    flat_labels: list[str] = []
    label_to_group: Dict[str, str] = {}
    for group, labels in hierarchical_labels.items():
        for label in labels:
            flat_label = f"{group}.{label}"
            flat_labels.append(flat_label)
            label_to_group[flat_label] = group

    tokenizer, clf = _load_pipeline(model_id)
    results = clf(texts, candidate_labels=flat_labels, multi_label=False)

    # Post-process results into TypeSafe format
    per_text_answers: List[Dict[str, Dict[str, Any]]] = []

    for res in results:
        raw_scores = [float(s) for s in res["scores"]]
        group_rescales: Dict[str, list] = {}
        for group, labels in hierarchical_labels.items():
            group_flat = [f"{group}.{label}" for label in labels]
            group_indices = [flat_labels.index(l) for l in group_flat]
            group_rescales[group] = [raw_scores[j] for j in group_indices]

        text_answers: Dict[str, Dict[str, Any]] = {}
        for name, meta in question_meta.items():
            q_type = meta["type"]
            group_probs = group_rescales[name]

            if q_type == "choice":
                flat_labels_for_group = [l for l in flat_labels if label_to_group[l] == name]
                choice_key = group_probs.index(max(group_probs))
                choice_label = flat_labels_for_group[choice_key].split(".")[-1]
                probs = {flat_labels_for_group[i].split(".")[-1]: float(group_probs[i]) for i in range(len(group_probs))}
                confidence = float(max(group_probs))
                text_answers[name] = {
                    "type": "choice",
                    "choice": choice_label,
                    "confidence": confidence,
                    "probabilities": probs,
                }
            elif q_type == "noul":
                yes_prob = group_probs[0]
                noul = 1.0 if yes_prob >= 0.5 else 0.0
                confidence = float(max(group_probs))
                text_answers[name] = {
                    "type": "noul",
                    "noul": noul,
                    "confidence": confidence,
                    "probabilities": {"yes": yes_prob, "no": 1 - yes_prob},
                }
            elif q_type == "score":
                sorted_indices = sorted(range(len(group_probs)), key=lambda j: group_probs[j], reverse=True)
                score = float(sum(j / len(hierarchical_labels[name]) for j in sorted_indices))
                probs = {str(i): float(group_probs[i]) for i in range(len(group_probs))}
                text_answers[name] = {
                    "type": "score",
                    "score": score,
                    "confidence": float(max(group_probs)),
                    "probabilities": probs,
                }

        per_text_answers.append(text_answers)

    return [{"model": "generic-zs", "answers": answers} for answers in per_text_answers]
