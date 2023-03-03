from transformers import pipeline
from config import device, tokenizer, model_type
from random import randint
from torch import randint as torch_rand


# Test some examples
fill_test_examples = [
    "Ana ide u <mask>.",
    "Osnovna <mask> Vuk Karadžić",
    "Kupio sam dva <mask> i mleko."
]

default_gen_input = ""


def fill_examples(mod, tok):
    # Create a Fill mask pipeline
    fill_mask = pipeline(
        "fill-mask",
        model=mod,
        tokenizer=tok,
        device=device,
        top_k=3
    )
    examples = []
    for example in fill_test_examples:
        examples.append([x["sequence"] for x in fill_mask(example)])
    return examples


def generate(model, context, length=20, temperature=0.75):

    encoded_input = context.to(device)
    output = model.generate(
        **encoded_input,
        bos_token_id=randint(1, 50000),
        do_sample=True,
        top_k=0,
        max_length=length,
        temperature=temperature,
        no_repeat_ngram_size=3,
        # top_p=0.95,
        num_return_sequences=1,
        pad_token_id=0
        )

    return output


def generatetion_test(mod, tok, samples=3, length=24, context=default_gen_input, temp=0.75):

    outs = []
    if context == "":
        tokens = torch_rand(low=260, high=52000, size=(1,))
        context = tok.decode(tokens, skip_special_tokens=True)

    context = tok(context, return_tensors="pt")
    cl = context.data["input_ids"].size()[1]

    for x in range(samples):
        output = generate(mod, context=context, length=length+cl, temperature=temp)

        decoded_output = []
        for sample in output:
            sample = sample[cl:]
            decoded_output.append(tokenizer.decode(sample, skip_special_tokens=True))

        outs.append("".join(decoded_output))

    return outs


def test(mod, tok=tokenizer):
    if model_type == "roberta":
        return fill_examples(mod, tok)
    elif model_type == "gpt2":
        return generatetion_test(mod, tok)

