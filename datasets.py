from torch.utils.data import Dataset
from json import load, dump, loads
from random import shuffle
from torch import tensor
from jsonlines import open as jlopen
from itertools import chain
from tqdm import tqdm


class TextualDataset(Dataset):
    def __init__(self, texts, tokenizer):
        print("encoding")
        self.examples = []
        for example in tqdm(texts, total=len(texts)):
            example = example.replace("\n\n", "")
            x = tokenizer.encode(example, add_special_tokens=False)
            self.examples += [x]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return tensor(self.examples[i])

    def __jdump__(self, path):
        print("saving")
        with open(path, "w") as jp:
            for x in tqdm(self.examples, total=self.__len__()):
                dump(x, jp)
                jp.write("\n")


class EncodedFiles2Dataset(Dataset):
    def __init__(self, path, files, shfl=True, trim=None, block=None, shfl_files=False, eos=2):
        self.examples = []
        if shfl_files:
            shuffle(files)
        for file in files:
            with open(path + file, "r") as jf:
                if block:
                    self.examples += list(self.split(list(chain(*load(jf))), block))

                if not block and trim:
                    for x in tqdm(jf.readlines()):
                        example = []
                        buffer = []
                        for y in loads(x):
                            buffer.append(y)
                            if y == eos:
                                if len(buffer) > trim:
                                    if example:
                                        self.examples.append(example)
                                        example = []
                                    self.examples.append(buffer[:trim])
                                    buffer = []
                                elif (len(example) + len(buffer)) > trim:
                                    self.examples.append(example)
                                    example = buffer
                                    buffer = []
                                else:
                                    example += buffer
                                    buffer = []
                        if len(buffer) > trim:
                            if example:
                                self.examples.append(example)
                            self.examples.append(buffer[:trim])
                        elif (len(example) + len(buffer)) > trim:
                            self.examples.append(example)
                            self.examples.append(buffer)
                        else:
                            example += buffer
                            if example:
                                self.examples.append(example)
    
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
        with open(path, "w") as jp:
            dump(self.examples, jp)

    def __jdumpwsplit__(self, path, dev_ratio=0.01, name=""):
        split_line = round(self.__len__() * dev_ratio)
        with jlopen(path + "dev" + name + ".jsonl", "w") as jp:
            jp.write_all(self.examples[:split_line])
        with jlopen(path + "train" + name + ".jsonl", "w") as jp:
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
