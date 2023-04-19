from datasets import TextualDataset, EncodedFiles2Dataset
from json import load
from os import listdir
from config import tokenizer, encoded_file_keyword
from tqdm import tqdm


def json2dataset(path, file, tokenizer, save=False, attr="sents", save_path=None):
    with open(path+file, "r", encoding="utf-8") as jf:
        sents = load(jf)[attr]

    dataset = TextualDataset(sents, tokenizer)
    if save:
        if not save_path:
            save_path = path + encoded_file_keyword + file
        dataset.__jdump__(save_path)
    else:
        return dataset


def json2sents(path, attr="sents"):
    with open(path, "r", encoding="utf-8") as jf:
        return load(jf)[attr]


def multipleJson2dataset(path):
    files = [x for x in listdir(path) if encoded_file_keyword not in x and ".json" in x]
    for file in tqdm(files, total=len(files)):
        json2dataset(path, file, tokenizer, save=True)


def encoded2datasets(path, files, trim=None, block=None, dev_ratio=0.1,
                     shfl=False, save=False, save_path=None, name=""):

    dataset = EncodedFiles2Dataset(path, files, shfl, trim=trim, block=block)
    if save:
        if save_path is None:
            save_path = path
        dataset.__jdumpwsplit__(save_path, dev_ratio, name=name)
    else:
        return dataset


def multipleEncoded2datasets(path, trim=None, block=None, shfl=False, name=""):
    files = [x for x in listdir(path) if encoded_file_keyword in x]
    encoded2datasets(path, files, save=True, trim=trim, block=block, shfl=shfl, name=name)


path_to_files = "C:/gpt2/korpusi/tajno/"
multipleJson2dataset(path_to_files)
multipleEncoded2datasets(path_to_files, trim=512, name="_bert-p")
multipleEncoded2datasets(path_to_files, block=128, name="_gpt-p")
