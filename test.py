from transformers import pipeline
from config import model_folder, tokenizer

# Create a Fill mask pipeline
fill_mask = pipeline(
    "fill-mask",
    model=model_folder,
    tokenizer=tokenizer
)
# Test some examples
test_examples = [
    "Ana ide u <mask>.",
    "Osnovna <mask> Vuk Karadžić",
    "Kupio sam dva <mask> i mleko."
]

for example in test_examples:
    print(fill_mask(example))