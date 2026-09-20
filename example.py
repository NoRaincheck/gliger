from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# model_id = "knowledgator/gliclass-large-v1.0"
model_id = "knowledgator/gliclass-small-v1.0"

model = GLiClassModel.from_pretrained(model_id)
model = GLiClassModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type="single-label"
)

hierarchical_labels = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["product", "service", "shipping"],
}

examples = [
    {
        "text": "Love this item, great quality!",
        "labels": {"sentiment": "positive", "topic": "product"},
        # "labels": ["sentiment.positive", "topic.product"]
    },
    {
        "text": "Customer support was unhelpful",
        "labels": {"sentiment": "negative", "topic": "service"},
        # "labels": ["sentiment.negative", "topic.service"],
    },
]

text = "The product quality is amazing but delivery was slow"


results = pipeline(
    text,
    hierarchical_labels,
    threshold=0.5,
    examples=examples,
    return_hierarchical=True,
)[0]
print(results)
