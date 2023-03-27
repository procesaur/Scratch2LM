from transformers import RobertaConfig, GPT2Config, GPTJConfig
from transformers import AutoModelWithLMHead, RobertaForMaskedLM, GPT2LMHeadModel, GPTJModel
from transformers import RobertaTokenizerFast, GPT2TokenizerFast, DataCollatorForLanguageModeling, TrainingArguments
from torch import cuda
from json import load


# # Settings # #
with open("training-congifs/config-config.json", "r") as cf:
    cfg = load(cf)

# paths
train_path = cfg["paths"]["train_path"].replace("%main_path%", cfg["paths"]["main_path"])
dev_path = cfg["paths"]["dev_path"].replace("%main_path%", cfg["paths"]["main_path"])
tokenizer_path = cfg["paths"]["tokenizer_path"].replace("%main_path%", cfg["paths"]["main_path"])
model_folder = cfg["paths"]["model_folder"].replace("%main_path%", cfg["paths"]["main_path"])

# model and training parameters
model_type = cfg["training-options"]["model"]  # gpt2 roberta
pretrained_model = cfg["training-options"]["pretrained"]
resume = cfg["training-options"]["resume-from-checkpoint"]
output_from_model = cfg["training-options"]["output_from_model"]

# Training args fill
training_args = TrainingArguments(
    output_dir=model_folder,
    overwrite_output_dir=True,
    evaluation_strategy='epoch',

    num_train_epochs=cfg["training-options"]["epochs"],
    per_device_train_batch_size=cfg["training-options"]["batch_size"],
    per_device_eval_batch_size=cfg["training-options"]["batch_size"],
    save_steps=cfg["training-options"]["save_steps"],
    save_total_limit=cfg["training-options"]["save_total"],
    learning_rate=cfg["training-options"]["learning_rate"],
    weight_decay=cfg["training-options"]["decay"],
    warmup_steps=cfg["training-options"]["warmup_steps"],
    remove_unused_columns=False
)

# misc
encoded_file_keyword = cfg["misc"]["encoded_file_keyword"]
device = "cuda:0" if cuda.is_available() else "cpu"

with open("training-congifs/" + model_type + ".json", "r") as mf:
    model_params = load(mf)


# tokenizer dependent
bos_token = 50259
eos_token = 50260

# Model initialization
if pretrained_model:
    model = AutoModelWithLMHead.from_pretrained(pretrained_model)
else:
    # Model configuration
    if "roberta" in model_type:
        tokenizer = RobertaTokenizerFast(tokenizer_file=tokenizer_path,
                                         pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")

        data_collator = DataCollatorForLanguageModeling(
            mlm=True,
            mlm_probability=0.15,
            tokenizer=tokenizer,
        )

        model_config = RobertaConfig(**model_params)

    elif "gpt" in model_type:
        tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path, padding=False, pad_token="a")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        if "gpt2" in model_type:
            model_config = GPT2Config(**model_params)
        elif "gptj" in model_type:
            model_config = GPTJConfig(**model_params)

    model_config.vocab_size = tokenizer.vocab_size
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.bos_token_id

    if "roberta" in model_type:
        model = RobertaForMaskedLM(config=model_config)
    elif "gpt2" in model_type:
        model = GPT2LMHeadModel.from_config(model_config)
    elif "gptj" in model_type:
        model = GPTJModel.from_config(model_config)


