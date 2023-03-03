# scratch2lm
Training transformer models (e.g. RoBERTa and GPT2-large) from scratch.

inspiration : https://huggingface.co/blog/how-to-train

## 1. TOKENIZER TRAINING (skip if you have tokenizer you want to use)
1. set **tokenizer_train_path** in **config.py:10**. It should point to a folder containing textual files.
2. lounch **train_tokenizer.py**. New tokenizer will be saved as **tokenizer.json** in the previously provided path.

## 2. DATA ENCODING
1. prepare the dataset as follows:
    - dataset should consist of json files
    - each json file should have **sents** list in the object root
    - **sents** list should consist of textual sentences
    - json file should look at least like so:
    ```
    {"sents" = ["Hello world.", "Are you doing OK?"]}
    ```
2. Ensure the correct path to your tokenizer, **tokenizer_path** is set correctly in **config.py:13**
3. use **encode_data.py** **multipleJson2dataset** method and provide it with the path to a directory containing your dataset files (json). 
    ```
    from encode_data import multipleJson2dataset
    multipleJson2dataset("path/to/your/files")
    ```
    This will use your set tokenizer to tokenize each sentence in the **sents** list and save it to a new json file, which will encompas your encoded data.
    You can recognize the new files by a keyword **_encoded_** witch you can change by changing the **encoded_file_keyword**  in **config.py:36**.
## 3. TRAINING SETS PREPARATION
1. use **encode_data.py** **multipleEncoded2datasets** method and provide it with the path to a directory containing your new dataset files (json). They will be filtered by the formentioned keyword. 
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
    If this is not what you want, you can edit default params for **encoded2datasets** function or edit the call to it from **multipleEncoded2datasets** (in **encode_data.py**).
## 4. MODEL TRAINING
1. Make sure that the parameters are set correctly in the **config.py**, namely: (If you hadn't changed any of the params so far, you shouldn't need to change that at the moment)
    - path to train and dev datasets: **train_path** and **dev_path** at **config.py:11-12**
    - path to the tokenizer you used for data encoding: **tokenizer_path** atoutput_from_model
    - model type you want to train: **model_type** at **config.py:16**, which should be "roberta" or "gpt2"
    - training parameters you want to use at **config.py:20-29**
    - **bos_token** and **eos_token** at **config.py:32-33**
    - adequate model parameters at **config.py:39-65** for gpt-2 and **config.py:68-83** for roberta
2. (optional) Set a list of sentences for masking fill test for BERT at by editing **fill_test_examples** at **test.py:7** or edit the defualt generation query for GPT by editing **default_gen_input** at **test.py:14**.
If you don't want this kind of output during the training, set **output_from_model** to False at **config.py:85**
3. Run **train.py**
    
