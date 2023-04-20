from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.models import BPE, WordPiece
from os import listdir, path as px
from encode_data import json2sents
from tqdm import tqdm
from json import load


def load_configs(cfg=None, cfgpath="training-congifs/config.json"):
    if not cfg:
        with open(cfgpath, "r", encoding="utf-8") as cf:
            cfg = load(cf)
    return cfg["tokenizer_training"]


cfg = load_configs()


def get_sents(files):
    print("files: " + str(len(files)))
    sentences = []
    for file in tqdm(files):
        try:
            sentences += json2sents(file)
        except:
            print(file)
    print("sentences: " + str(len(sentences)))


def train_a_tokenizer(path=cfg["path"]):
    special_tokens = cfg["special_tokens"]
    unk_token = cfg["unk_token"]

    # Initialize a tokenizer
    if cfg["type"] == "BPE":
        tokenizer = Tokenizer(BPE(unk_token=unk_token))

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        tokenizer_trainer = BpeTrainer(
            vocab_size=cfg["size"],
            min_frequency=cfg["freq"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
            special_tokens=special_tokens
        )

    else:
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        tokenizer_trainer = WordPieceTrainer(
            vocab_size=cfg["size"],
            min_frequency=cfg["freq"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
            special_tokens=special_tokens,
        )

    files = [path + x for x in listdir(path) if px.isfile(path + x) and x.endswith(".json")]
    tokenizer.train_from_iterator(get_sents(files), tokenizer_trainer)
    tokenizer.save(path + "tokenizer.json", pretty=True)


train_a_tokenizer()
