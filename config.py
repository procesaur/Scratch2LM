from transformers import TrainingArguments, AutoConfig, RobertaConfig, GPT2Config, GPTJConfig, T5Config
from transformers import RobertaForMaskedLM, GPT2LMHeadModel, GPTJModel, T5Model
from transformers import RobertaTokenizerFast, AutoTokenizer, T5TokenizerFast
from transformers import DataCollatorForTokenClassification, DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from torch import cuda
from json import load
from os import path as px, chdir

chdir(px.dirname(px.abspath(__file__)))

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
            with open("training-configs/" + model_type + ".json", "r", encoding="utf-8") as mf:
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


def load_tokenizer(model_type, tokenizer_path, tuning=False):
    if "roberta" in model_type:
        if tuning:
            return RobertaTokenizerFast(tokenizer_file=tokenizer_path, add_prefix_space=True, max_len=514,
                                        pad_token="<pad>", unk_token="<unk>", mask_token="<mask>", pad_to_max_length=True)
        else:
            return RobertaTokenizerFast(tokenizer_file=tokenizer_path,
                                        pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    elif "gpt" in model_type:
        if tuning:
            return RobertaTokenizerFast(tokenizer_file=tokenizer_path, add_prefix_space=True,
                                        padding=False, pad_token="<pad>")
        else:
            return RobertaTokenizerFast(tokenizer_file=tokenizer_path,
                                        padding=False, pad_token="<pad>")
    else:
        return AutoTokenizer()


def collator(model_type, fast_tokenizer):
    if "roberta" in model_type:
        return DataCollatorForWholeWordMask(
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


def tune_collator(tknzr, task=""):
    if "token" in task:
        return DataCollatorForTokenClassification(tokenizer=tknzr, padding=True)


def load_configs(cfg=None, cfgpath="training-configs/config.json", tuning=False):
    # cfgpath = px.dirname(px.abspath(__file__)) + "/" + cfgpath
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
    if not tuning:
        efk = cfg["misc"]["encoded_file_keyword"]
        default_input = cfg["misc"]["default_gen_input"]

        return newpaths, options, args, efk, default_input, tokenizer_training
    else:
        return newpaths, options, args


def process_path(path, key, replace_path):
    if isinstance(path, str):
        return path.replace(key, replace_path)
    else:
        results = []
        for x in path:
            results.append(x.replace(key, replace_path))
        return results


def get_examples(examples=None, examples_path="training-configs/fill_mask_examples.json"):
    if not examples:
        with open(examples_path, "r", encoding="utf-8") as ef:
            examples = load(ef)
    return examples


paths, model_options, training_args, encoded_file_keyword, default_gen_input, tokenizer_training = load_configs()
fill_test_examples = get_examples()
tokenizer = load_tokenizer(model_options["model_type"], paths["tokenizer_path"])

data_collator = collator(model_options["model_type"], tokenizer)
model = get_model(model_options["model_type"], tokenizer, paths["pretrained"])
device = "cuda:0" if cuda.is_available() else "cpu"


tuning_paths, tuning_options, tuning_args = load_configs(cfgpath="training-configs/tuning_config.json", tuning=True)
tuning_tokenizer = load_tokenizer(tuning_options["model_type"], tuning_paths["tokenizer_path"], tuning=True)
tuning_collator = tune_collator(tuning_tokenizer, tuning_options["tuning_task"])
