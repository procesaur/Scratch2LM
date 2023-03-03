from os import path as px
from transformers import RobertaConfig, GPT2Config, AutoModelWithLMHead, RobertaForMaskedLM, GPT2LMHeadModel
from transformers import RobertaTokenizerFast, GPT2TokenizerFast, DataCollatorForLanguageModeling, TrainingArguments
from torch import cuda


# # Settings # #

# paths
tokenizer_train_path = "C:/gpt2/za_tokenizer/"
train_path = "data/mini_train.json"
dev_path = "data/mini_dev.json"
tokenizer_path = "data/tokenizer.json"

# model
model_type = "roberta"  # gpt2 roberta
pretrained_model = None

# training parameters
model_folder = "saved"
epochs = 3
learning_rate = 0.0001
decay = 0.01
batch_size = 1
dev_batch_size = 8
save_steps = 128  # 8192
eval_steps = 128  # 4096
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
        num_attention_heads=8,  # 16
        num_hidden_layers=12,  # 24
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

# Paths computation
train_path = px.join(px.dirname(__file__), train_path)
dev_path = px.join(px.dirname(__file__), dev_path)
tokenizer_path = px.join(px.dirname(__file__), tokenizer_path)
model_folder = px.join(px.dirname(__file__), model_folder)

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
