from config import tuning_paths, tuning_collator, tuning_args, tuning_options
import evaluate
import numpy as np
from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer, RobertaTokenizerFast
from datasets import load_dataset


x_models = [
     tuning_paths["pretrained"]
]

for x_model in x_models:
    tuning_tokenizer = RobertaTokenizerFast.from_pretrained(x_model, add_prefix_space=True, pad_to_max_length=True,
                                                            pad_token="<pad>", unk_token="<unk>", mask_token="<mask>",
                                                            max_len=256)
    # tuning_tokenizer = AutoTokenizer.from_pretrained(x_model)

    label2id = {
        "ADJ": 0,
        "ADP": 1,
        "PUNCT": 2,
        "ADV": 3,
        "AUX": 4,
        "SYM": 5,
        "INTJ": 6,
        "CCONJ": 7,
        "X": 8,
        "NOUN": 9,
        "DET": 10,
        "PROPN": 11,
        "NUM": 12,
        "VERB": 13,
        "PART": 14,
        "PRON": 15,
        "SCONJ": 16
    }

    label2id = {'I-LOC': 0, 'B-PERS': 1, 'B-EVENT': 2, 'I-ORG': 3, 'B-LOC': 4, 'B-DEMO': 5, 'I-DEMO': 6, 'I-PERS': 7,
         'I-WORK': 8, 'I-ROLE': 9, 'B-WORK': 10, 'I-EVENT': 11, 'B-ROLE': 12, 'B-ORG': 13, 'O': 14}

    label_list = list(label2id.keys())
    id2label = {v: k for k, v in label2id.items()}


    def compute_metrics(p):

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    def tokenize_and_align_labels(examples):
        tokenized_inputs = tuning_tokenizer(examples[tuning_options["token_col"]], truncation=True, is_split_into_words=True, padding=True)
        labels = []
        for i, label in enumerate([map_labels(x, label2id) for x in examples[tuning_options["label_col"]]]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def map_labels(labels, label_dict):
        results = []
        for label in labels:
            if label in label_dict:
                results.append(label_dict[label])
            else:
                results.append(-100)
        return results


    tuning_dataset = load_dataset('json', data_files=tuning_paths["train_path"])
    original_columns = tuning_dataset["train"].column_names
    tuning_dataset = tuning_dataset["train"].train_test_split(test_size=0.1, seed=0)
    tuning_dataset = tuning_dataset.map(tokenize_and_align_labels, batched=True)

    # eval_dataset = load_dataset('json', data_files=tuning_paths["dev_path"])
    # eval_dataset = tuning_dataset.map(tokenize_and_align_labels, batched=True)
    eval_dataset = tuning_dataset["test"]
    tuning_dataset = tuning_dataset["train"].remove_columns(original_columns)
    eval_dataset = eval_dataset.remove_columns(original_columns)

    seqeval = evaluate.load("seqeval")

    model = AutoModelForTokenClassification.from_pretrained(x_model,
                                                            num_labels=len(label2id), id2label=id2label, label2id=label2id)

    trainer = Trainer(
        model=model,
        args=tuning_args,
        train_dataset=tuning_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tuning_tokenizer,
        data_collator=tuning_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
