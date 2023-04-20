from torch.utils.data import Dataset
from json import load, dump, loads
from random import shuffle
from torch import tensor
from jsonlines import open as jlopen
from itertools import chain


class TextualDataset(Dataset):
    def __init__(self, sents, tokenizer):
        self.examples = []
        for example in sents:
            x = tokenizer.encode(example)
            self.examples += [x]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return tensor(self.examples[i])

    def __jdump__(self, path):
        with open(path, "w", encoding="utf-8") as jp:
            dump(self.examples, jp)


class EncodedFiles2Dataset(Dataset):
    def __init__(self, path, files, shfl=True, trim=None, block=None):
        self.examples = []
        for file in files:
            with open(path + file, "r", encoding="utf-8") as jf:
                if block:
                    self.examples += list(self.split(list(chain(*load(jf))), block))
                if not block and trim:
                    self.examples += [x[:trim] for x in load(jf)]
                if not block and not trim:
                    self.examples += load(jf)
        if shfl:
            shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def split(self, list_a, chunk_size):
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]

    def __getitem__(self, i):
        return tensor(self.examples[i])

    def __jdump__(self, path):
        with open(path, "w", encoding="utf-8") as jp:
            dump(self.examples, jp)

    def __jdumpwsplit__(self, path, dev_ratio=0.1):
        split_line = round(self.__len__() * dev_ratio)
        with jlopen(path + "dev.jsonl", "w", encoding="utf-8") as jp:
            jp.write_all(self.examples[:split_line])
        with jlopen(path + "train.jsonl", "w", encoding="utf-8") as jp:
            jp.write_all(self.examples[split_line:])


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

