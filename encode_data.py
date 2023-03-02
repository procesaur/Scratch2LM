from datasets import TextualDataset, EncodedFiles2Dataset
from json import load
from os import listdir
from config import tokenizer, encoded_file_keyword


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


def multipleJson2dataset(path):
    for file in [x for x in listdir(path) if encoded_file_keyword not in x and ".json" in x]:
        print(file)
        json2dataset(path, file, tokenizer, save=True)


def encoded2datasets(path, files, dev_ratio=0.1, shfl=True, save=False, save_path=None):

    if save_path is None:
        save_path = path

    dataset = EncodedFiles2Dataset(path, files, shfl)
    if save:
        dataset.__jdumpwsplit__(save_path, dev_ratio)
    else:
        return dataset


def multipleEncoded2datasets(path):
    files = [x for x in listdir(path) if encoded_file_keyword in x]
    encoded2datasets(path, files[2:4], save=True)


# multipleEncoded2datasets(path)
# json2dataset(path, file, tokenizer, save=True)
