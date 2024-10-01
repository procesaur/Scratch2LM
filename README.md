# scratch2lm
Training transformer models (e.g. RoBERTa and GPT2-large) from scratch.

inspiration : https://huggingface.co/blog/how-to-train


## 1. TOKENIZER TRAINING (skip if you have tokenizer you want to use)
1. set **tokenizer_path** in [config.json](training-configs/config.json). It should point to a folder containing textual files.
2. launch [train_tokenizer.py](train_tokenizer.py). New tokenizer will be saved as **tokenizer.json** in the previously provided path.

## 2. DATA ENCODING
1. prepare the dataset as follows:
    - dataset should consist of jsonl files
    - each json line should have **text** string in the object root
    - each line should look like e.g.:
    ```
{"id": "12", "text": "<s>UTF-8 варијанта је најзгоднија за кодирање већински латиничног текста.</s><s>Дато је и кратко упутство..."}
    ```
2. Ensure the correct path to your tokenizer, **tokenizer_path** is set correctly in [config.json](training-configs/config.json)
3. use [encode_data.py](encode_data.py) **multipleJson2dataset** method and provide it with the path to a directory containing your dataset files (json). 
    ```
    from encode_data import multipleJson2dataset
    multipleJson2dataset("path/to/your/files")
    ```
    This will use your set tokenizer to tokenize each sentence in the **sents** list and save it to a new **jsonl** file, which will encompas your encoded data.
    You can recognize the new files by a keyword **_encoded_** witch you can change by changing the **encoded_file_keyword**  in [config.json](training-configs/config.json).
## 3. TRAINING SETS PREPARATION
1. use [encode_data.py](encode_data.py) **multipleEncoded2datasets** method and provide it with the path to a directory containing your new dataset files (json). They will be filtered by the formentioned keyword. 
If you hadn't mendled with the settings after the second step, just supply it with the same path. 

    If you are training a BERT-based model you should probably pass it **trim** arg that will trim each sentence to that size.
    ```
    from encode_data import multipleEncoded2datasets
    multipleEncoded2datasets("path/to/your/files", trim=512)
    ```
    
    If you are training a GPT-based model you should probably pass it **block** arg that will block text into chunks of the size.
    ```
    from encode_data import multipleEncoded2datasets
    multipleEncoded2datasets("path/to/your/files", block=512)
    ```
    This will, by default, combine all of your data into one list, randomly shuffle it, split to training and dev set in 9:1 ratio and save it one the same path with names **train.json** and **dev.json**.
    If this is not what you want, you can edit default params for **encoded2datasets** function or edit the call to it from **multipleEncoded2datasets** (in [encode_data.py](encode_data.py)).
## 4. MODEL TRAINING
1. Make sure that the parameters are set correctly in the [config.json](training-configs/config.json), namely: (If you hadn't changed any of the params so far, you shouldn't need to change that at the moment)
    - path to your tokenizer, train and dev datasets: **tokenizer_path**, **train_path** and **dev_path** in **paths** section of the [config.json](training-configs/config.json)
      as well as your **model_folder**, especially if you are continuing from a checkpoint
    - in the same section you configure whether you are using a pretrained model by assigning a path (**pretrained**)
    - model type you want to train: **model_type** in [config.json](training-configs/config.json), which should be one of the currently avaialble, and adequate model parameters can be found and adjusted in the [training config](training-configs) folder.
         - gpt2-large
         - roberta-base
         - roberta-large
         - gptj
    - in the same section you can configure if you want to resume from previous checkpoints (**resume-from-checkpoint**), and if you want to log model tests along the training (**output_from_modelt**)
    - training parameters you want to use are set in the training-options section of [config.json](training-configs/config.json)

2. (optional, if you selected **true** for **output_from_modelt**) Set a list of sentences for masking fill test for BERT at by editing [fill_mask_examples.json](training-configs/fill_mask_examples.json) or edit the defualt generation query for GPT by editing **default_gen_input** in the misc section of [config.json](training-configs/config.json).

3. Run [train.py](train.py)
    
## REMARKS

(training) Code is also available as a jupyter notebook in the [note.ipynb](note.ipynb) file, and as single python file in [bundle.py](bundle.py). In this case, all configs are contained and should be edited within.