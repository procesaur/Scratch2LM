from transformers import pipeline
from config import device, tokenizer


# Test some examples
test_examples = [
    "Ana ide u <mask>.",
    "Osnovna <mask> Vuk Karadžić",
    "Kupio sam dva <mask> i mleko."
]


def fill_examples(mod, tok=tokenizer):
    # Create a Fill mask pipeline
    fill_mask = pipeline(
        "fill-mask",
        model=mod,
        tokenizer=tok,
        device=device,
        top_k=3
    )

    examples = []
    for example in test_examples:
        examples.append([x["sequence"] for x in fill_mask(example)])
    return examples



