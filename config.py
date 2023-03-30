from transformers import TrainingArguments, AutoConfig, RobertaConfig, GPT2Config, GPTJConfig
from transformers import AutoModelWithLMHead, RobertaForMaskedLM, GPT2LMHeadModel, GPTJModel
from transformers import RobertaTokenizerFast, GPT2TokenizerFast, AutoTokenizer, DataCollatorForLanguageModeling
from torch import cuda
from json import load


def get_model(model_type, fast_tokenizer, model_params=None, pretrained=""):
    if pretrained:
        return AutoModelWithLMHead.from_pretrained(pretrained)
    else:
        if not model_params:
            with open("training-congifs/" + model_type + ".json", "r") as mf:
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
        with open(cfgpath, "r") as cf:
            cfg = load(cf)

    # paths
    main_path = cfg["paths"]["main_path"]
    paths = {x: process_path(y, "%main_path%", main_path) for (x, y) in cfg["paths"].items()}

    # model and training parameters
    model_options = cfg["model-options"]

    training_options = cfg["training-options"]
    training_options["output_dir"] = paths["model_folder"]
    training_options["remove_unused_columns"] = False

    # Training args fill
    training_args = TrainingArguments(**training_options)
    encoded_file_keyword = cfg["misc"]["encoded_file_keyword"]
    default_gen_input = cfg["misc"]["default_gen_input"]
    return paths, model_options, training_args, encoded_file_keyword, default_gen_input


def process_path(path, key, replace_path):
    if isinstance(path, str):
        return path.replace(key, replace_path)
    else:
        results = []
        for x in path:
            results.append(path.replace(x, replace_path))
        return results


def get_examples(examples=None, examples_path="training-congifs/fill_mask_examples.json"):
    if not examples:
        with open(examples_path, "r") as ef:
            examples = load(ef)
    return examples


paths, model_options, training_args, encoded_file_keyword, default_gen_input = load_configs()
fill_test_examples = get_examples()
tokenizer = load_tokenizer(model_options["model_type"], paths["tokenizer_path"])
data_collator = collator(model_options["model_type"], tokenizer)
model = get_model(model_options["model_type"], tokenizer)
device = "cuda:0" if cuda.is_available() else "cpu"
