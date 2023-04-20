from json import loads, load
from random import randint
from torch.utils.data import Dataset
from torch import tensor, cuda, randint as torch_rand, save

from transformers import TrainingArguments, AutoConfig, RobertaConfig, GPT2Config, GPTJConfig
from transformers import RobertaForMaskedLM, GPT2LMHeadModel, GPTJModel
from transformers import RobertaTokenizerFast, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import DefaultFlowCallback
from transformers.trainer_callback import TrainerState, TrainerControl, TrainingArguments, IntervalStrategy
from transformers import pipeline, Trainer

for c in range(0, cuda.device_count()):
    print(cuda.get_device_name(c))


class JsonDataset(Dataset):
    def __init__(self, jpath):
        if isinstance(jpath, str):
            with open(jpath, "r", encoding="utf-8") as jf:
                self.examples = list(jf)
        else:
            self.examples = []
            for jp in jpath:
                with open(jp, "r", encoding="utf-8") as jf:
                    self.examples += list(jf)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return tensor(loads(self.examples[i])).long()


false = False
true = True

config = {
  "paths": {
    "main_path": "data",
    "train_path": "%main_path%/train_mini_bert.jsonl",
    "dev_path": "%main_path%/dev_mini_bert.jsonl",
    "tokenizer_path":"%main_path%/tokenizer.json",
    "model_folder": "%main_path%/saved",
    "pretrained": ""
  },

  "model-options": {
    "model_type": "roberta-base",
    "resume-from-checkpoint": false,
    "output_from_model": true
  },

  "training-options": {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 0.00005,
    "weight_decay": 0,
    "warmup_steps": 2000,

    "save_steps": 50000,
    "eval_steps": 50000,
    "save_total_limit": 1,
    "load_best_model_at_end": false,
    "overwrite_output_dir": true,
    "evaluation_strategy": "epoch"
  },

  "misc": {
      "encoded_file_keyword": "_encoded_",
      "default_gen_input": ""
  },

  "tokenizer_training": {
    "path" : "C:/gpt2/tokenizer/",
    "size": 49152,
    "freq": 2,
    "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    "unk_token": "<UNK>",
    "type": "BPE"
  }
}

examples = [
    "Ana ide u <mask>.",
    "Osnovna <mask> Vuk Karadžić",
    "Kupio sam dva <mask> i mleko."
]


def get_model(model_type, fast_tokenizer, pretrained="", model_params=None):
    if pretrained:
        if "roberta" in model_type:
            return RobertaForMaskedLM.from_pretrained(pretrained)
        elif "gpt2" in model_type:
            return GPT2LMHeadModel.from_pretrained(pretrained)
        elif "gptj" in model_type:
            return GPTJModel.from_pretrained(pretrained)

    else:
        if not model_params:
            with open("training-congifs/" + model_type + ".json", "r", encoding="utf-8") as mf:
                model_params = load(mf)
        return create_model(model_type, fast_tokenizer, model_params)


def create_model(model_type, fast_tokenizer, model_params):
    if "roberta" in model_type:
        model_config = RobertaConfig(**model_params)
    elif "gpt2" in model_type:
        model_config = GPT2Config(**model_params)
    elif "gptj" in model_type:
        model_config = GPTJConfig(**model_params)

    else:
        model_config = AutoConfig()

    model_config.vocab_size = fast_tokenizer.vocab_size
    model_config.bos_token_id = fast_tokenizer.bos_token_id
    model_config.eos_token_id = fast_tokenizer.bos_token_id

    if "roberta" in model_type:
        return RobertaForMaskedLM(config=model_config)
    elif "gpt2" in model_type:
        return GPT2LMHeadModel(config=model_config)
    elif "gptj" in model_type:
        return GPTJModel(config=model_config)


def load_tokenizer(model_type, tokenizer_path):
    if "roberta" in model_type:
        return RobertaTokenizerFast(tokenizer_file=tokenizer_path,
                                    pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    elif "gpt" in model_type:
        return RobertaTokenizerFast(tokenizer_file=tokenizer_path, padding=False, pad_token="<pad>")
    else:
        return AutoTokenizer()


def collator(model_type, fast_tokenizer):
    if "roberta" in model_type:
        return DataCollatorForLanguageModeling(
            mlm=True,
            mlm_probability=0.15,
            tokenizer=fast_tokenizer,
        )
    elif "gpt" in model_type:
        return DataCollatorForLanguageModeling(
            tokenizer=fast_tokenizer,
            mlm=False,
        )
    else:
        return DataCollatorForLanguageModeling(
            tokenizer=fast_tokenizer
        )


def load_configs(cfg=None, cfgpath="training-congifs/config.json"):
    if not cfg:
        with open(cfgpath, "r", encoding="utf-8") as cf:
            cfg = load(cf)

    # paths
    main_path = cfg["paths"]["main_path"]
    newpaths = {x: process_path(y, "%main_path%", main_path) for (x, y) in cfg["paths"].items()}

    # model and training parameters
    options = cfg["model-options"]

    training_options = cfg["training-options"]
    training_options["output_dir"] = newpaths["model_folder"]
    training_options["remove_unused_columns"] = False
    tokenizer_training = cfg["tokenizer_training"]

    # Training args fill
    args = TrainingArguments(**training_options)
    efk = cfg["misc"]["encoded_file_keyword"]
    default_input = cfg["misc"]["default_gen_input"]

    return newpaths, options, args, efk, default_input, tokenizer_training


def process_path(path, key, replace_path):
    if isinstance(path, str):
        return path.replace(key, replace_path)
    else:
        results = []
        for x in path:
            results.append(x.replace(key, replace_path))
        return results


def get_examples(examples=None, examples_path="training-congifs/fill_mask_examples.json"):
    if not examples:
        with open(examples_path, "r", encoding="utf-8") as ef:
            examples = load(ef)
    return examples


paths, model_options, training_args, encoded_file_keyword, default_gen_input, tokenizer_training = load_configs(config)
fill_test_examples = get_examples()
tokenizer = load_tokenizer(model_options["model_type"], paths["tokenizer_path"])
data_collator = collator(model_options["model_type"], tokenizer)
model = get_model(model_options["model_type"], tokenizer, paths["pretrained"])
device = "cuda:0" if cuda.is_available() else "cpu"


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
    if "roberta" in model_options["model_type"]:
        return fill_examples(mod, tok)
    elif "gpt" in model_options["model_type"]:
        return generatetion_test(mod, tok)


class CustomDefaultFlowCallback(DefaultFlowCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        # Save model?
        if model_options["save_each_epoch"]:
            save(kwargs["model"], paths["model_folder"] + "/epoch_" + str(state.epoch))

        return control

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

            # Perform Experiment?
            if model_options["output_from_model"]:
                examples = test(kwargs["model"])
                if not isinstance(examples[0], str):
                    examples = [e for ee in examples for e in ee]
                with open(paths["model_folder"] + "/experiments.log", "a+", encoding="utf-8") as lf:
                    lf.write("\t".join(examples))
                    lf.write("\n")

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control


train_dataset = JsonDataset(paths["train_path"])
eval_dataset = JsonDataset(paths["dev_path"])

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # prediction_loss_only=True,
)

trainer.remove_callback(DefaultFlowCallback)
trainer.add_callback(CustomDefaultFlowCallback)

# Train the model
trainer.train(resume_from_checkpoint=model_options["resume-from-checkpoint"])
