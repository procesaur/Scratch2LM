#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install transformers')
# get_ipython().system('pip install --upgrade ipywidgets')

from torch.utils.data import Dataset
from json import loads
from torch import tensor

from transformers import RobertaConfig, GPT2Config, AutoModelWithLMHead, RobertaForMaskedLM, GPT2LMHeadModel
from transformers import RobertaTokenizerFast, GPT2TokenizerFast, DataCollatorForLanguageModeling, TrainingArguments
from torch import cuda
from transformers import DefaultFlowCallback, ProgressCallback
from transformers.trainer_callback import TrainerState, TrainerControl, TrainingArguments, IntervalStrategy

from transformers import pipeline, Trainer
from random import randint
from torch import randint as torch_rand


for c in range(0, cuda.device_count()):
    print(cuda.get_device_name(c))


class JsonDataset(Dataset):
    def __init__(self, jpath):
        with open(jpath, "r", encoding="utf-8") as jf:
            self.examples = list(jf)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return tensor(loads(self.examples[i]))


# # Settings # #

# paths
main_path = "C:/Users/Administrator/Desktop/training/"
train_path = main_path + "train.jsonl"
dev_path = main_path + "dev.jsonl"
tokenizer_path = main_path + "tokenizer.json"

# model and training parameters
model_type = "roberta"  # gpt2 roberta
pretrained_model = None
resume = True

model_folder = main_path + "saved"
epochs = 4
learning_rate = 0.0001
decay = 0.01
batch_size = 8
dev_batch_size = 8
save_steps = 8192  # 8192
eval_steps = 4096  # 4096
save_total_limit = 1
warmup_steps = 5  # 500

# tokenizer dependent
bos_token = 50259
eos_token = 50260

# misc
encoded_file_keyword = "_encoded_"

# model config for gpt2
gpt2_large_config = GPT2Config(
        attn_pdrop=0.1,
        bos_token_id=bos_token,
        embd_pdrop=0.1,
        eos_token_id=eos_token,
        initializer_range=0.02,
        layer_norm_epsilon=1e-05,
        model_type="gpt2",
        n_ctx=1024,
        n_embd=1280,
        n_head=20,
        n_layer=36,
        n_positions=1024,
        resid_pdrop=0.1,
        summary_activation=None,
        summary_first_dropout=0.1,
        summary_proj_to_labels=True,
        summary_type="cls_index",
        summary_use_proj=True,
        task_specific_params={
            "text-generation":
            {
              "do_sample": True,
              "max_length": 50
            }
        }
    )

# model config for roberta
roberta_large_config = RobertaConfig(
        max_position_embeddings=514,
        num_attention_heads=16,  # 16
        num_hidden_layers=24,  # 24
        type_vocab_size=1,

        attention_probs_dropout_prob=0.1,
        bos_token_id=bos_token,
        eos_token_id=eos_token,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        layer_norm_eps=1e-05,
    )

output_from_model = True

# Device initialization
device = "cuda:0" if cuda.is_available() else "cpu"

# Model initialization
if pretrained_model:
    model = AutoModelWithLMHead.from_pretrained(pretrained_model)
else:
    # Model configuration
    if model_type == "gpt2":

        tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path, padding=False, pad_token="a")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )

        model_config = gpt2_large_config
        model_config.vocab_size = tokenizer.vocab_size
        model = GPT2LMHeadModel.from_config(model_config)

    elif model_type == "roberta":

        tokenizer = RobertaTokenizerFast(tokenizer_file=tokenizer_path,
                                         pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")

        data_collator = DataCollatorForLanguageModeling(
            mlm=True,
            mlm_probability=0.15,
            tokenizer=tokenizer,
        )

        model_config = roberta_large_config
        model_config.vocab_size = tokenizer.vocab_size
        model = RobertaForMaskedLM(config=model_config)

# Training args fill
training_args = TrainingArguments(
    output_dir=model_folder,
    overwrite_output_dir=True,
    evaluation_strategy='epoch',
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    weight_decay=decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=dev_batch_size,
    save_steps=save_steps,
    eval_steps=eval_steps,
    save_total_limit=save_total_limit,
    warmup_steps=warmup_steps
)

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


class CustomDefaultFlowCallback(DefaultFlowCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % args.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True
            examples = test(kwargs["model"])
            examples = [e for ee in examples for e in ee]
            with open(model_folder + "/log", "a+", encoding="utf-8") as lf:

                lf.write("\t".join(examples))
                lf.write("\n")

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control


eval_dataset = JsonDataset(dev_path)
train_dataset = JsonDataset(train_path)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #prediction_loss_only=True,
)

if output_from_model:
    trainer.remove_callback(DefaultFlowCallback)
    trainer.add_callback(CustomDefaultFlowCallback)

trainer.train(resume_from_checkpoint=resume)
