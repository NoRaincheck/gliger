"""Example usage of GLiClass for hierarchical text classification.

This script demonstrates how to use the GLiClass model and ZeroShotClassificationPipeline
to classify text into hierarchical categories such as sentiment and topic.
"""

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# Model configuration
MODEL_ID = "knowledgator/gliclass-small-v1.0"


def load_model_and_pipeline(model_id: str) -> tuple[GLiClassModel, ZeroShotClassificationPipeline]:
    """Load the GLiClass model and tokenizer, then create a classification pipeline.

    Args:
        model_id: The Hugging Face model identifier to load.

    Returns:
        A tuple containing the loaded model and classification pipeline.
    """
    model = GLiClassModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipeline = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="single-label"
    )
    return model, pipeline


def main() -> None:
    """Run a demonstration of hierarchical text classification."""
    # Load model and pipeline
    model, pipeline = load_model_and_pipeline(MODEL_ID)

    # Define hierarchical label structure
    hierarchical_labels = {
        "sentiment": ["positive", "negative", "neutral"],
        "topic": ["product", "service", "shipping"],
    }

    # Example labeled data for few-shot learning
    examples = [
        {
            "text": "Love this item, great quality!",
            "labels": {"sentiment": "positive", "topic": "product"},
        },
        {
            "text": "Customer support was unhelpful",
            "labels": {"sentiment": "negative", "topic": "service"},
        },
    ]

    # Text to classify
    text = "The product quality is amazing but delivery was slow"

    # Perform classification
    results = pipeline(
        text,
        hierarchical_labels,
        threshold=0.5,
        examples=examples,
        return_hierarchical=True,
    )[0]

    print(results)


if __name__ == "__main__":
    main()
