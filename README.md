# GLiClass — TypeSafe API Primitives via Zero-Shot Classification

**gliger** emulates the [TypeSafe API primitives](https://docs.typesafe.ai/introduction/quickstart) — `Choice`, `Noul`, and `Score` — using HuggingFace zero-shot classification pipelines (GLiClass or NLI models).

## Quick Start

```bash
# Install dependencies
uv sync

# Run the example
uv run python example.py
```

## What It Does

| Primitive | Type | Output |
|-----------|------|--------|
| **Choice**  | Multi-class | Selected label + confidence + per-label probabilities |
| **Noul**    | Binary (yes/no) | `0.0` or `1.0` + confidence + yes/no probabilities |
| **Score**   | Ordinal (0..N) | Fractional score + confidence + legend + per-level probabilities |

Three modules are available depending on your model preference:

| Module | Backend | Best for |
|--------|---------|----------|
| `example.py` | GLiClass (`knowledgator/gliclass-large-v1.0`) | Fast, single-batch inference with hierarchical labels |
| `zs_clf.py` | NLI (`tasksource/ModernBERT-base-nli`) | JEV adapter via HuggingFace pipeline |
| `generic_zs_clf.py` | Model-agnostic | Drop-in zero-shot classification against any label set |

## Usage

### Choice (multi-class classification)

```python
from zs_clf import adapt_choice

results = adapt_choice(
    texts=["I need help with my Stripe integration"],
    hierarchical_labels={
        "department": ["billing", "technical", "sales"],
        "priority": ["low", "medium", "high"],
    },
)
```

### Noul (binary yes/no)

```python
from zs_clf import adapt_noul

results = adapt_noul(
    texts=["This is urgent, please help ASAP"],
    instructions="The message conveys urgency",
)
```

### Score (ordinal ranking)

```python
from zs_clf import adapt_score

results = adapt_score(
    texts=["I'm really frustrated with this product"],
    instructions="How angry the customer sounds",
    criteria=["Calm", "Frustrated", "Very angry"],
)
```

### Unified API (all primitives at once)

```python
from zs_clf import jev_api

results = jev_api(
    texts=["Hi, my Stripe integration keeps failing"],
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {"billing": "Payment issues", "technical": "Bugs"},
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency",
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer is",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
)
```

### Generic Zero-Shot Classification

```python
from generic_zs_clf import classify

results = classify(
    texts=["I need a refund"],
    candidate_labels=["billing", "technical", "sales", "general"],
    model_id="facebook/bart-large-mnli",  # or any HF zero-shot model
)
```

## Design Highlights

- **Single-pass inference** — all texts and labels processed in one forward pass
- **Per-group renormalization** — softmax applied within each question group for calibrated probabilities
- **Fractional ordinal scores** — Score uses rank-weighted blending for soft ordinal outputs
- **Deterministic Noul** — binary threshold at 0.5 with confidence = max(yes, no)
- **Hierarchical labels** — supports grouped labels via dot-notation (`group.label`)
- **Model caching** — pipelines are cached to avoid reloads

## Requirements

- Python 3.14+
- `uv` for environment management

## Development

```bash
# Lint
uv run ruff check .

# Type check
uv run ty check
```

## License

Private
