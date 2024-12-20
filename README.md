# Evaluating Tokenizers on Typologically Distinct Languages: Insights from Turkish and Spanish
Anders Hjulmand, Eisuke Okuda, Andreas Flensted

## 1. Introduction

In this study, we analyzed the characteristics and performance of BPE, WordPiece, and Unigram tokenizers on Turkish, an agglutinative language, and Spanish, a fusional language. We found that Turkish uses more tokens per word, had a longer average token length, and was more difficult to model.

Tinybert was trained with a Masked Language Modeling (MLM) objective by masking 15\% of tokens. We trained for 1 epoch with a peak learning rate of 5e-4, batch size of 64 and gradient accumulation of 8.

In our further analysis, we found that the BPE tokenizer with a vocabulary size of 40K aligned best with Turkish morphemes:

![hhghfghfghfgh](/figures/turkish_f1_morpheme_results.png)


## 2. Setup

1. **Clone repository**
   ```bash
    git clone https://github.com/ah140797/NLP-project.git
    ``` 

2. **Install the Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **WandB Login:**

    To use `wandb` for experiment tracking, you need to login:

    ```bash
    wandb login
    ```

    This will prompt you to enter an API key from wandb.
    <br>
    

4. **HuggingFace Login:**
   
    To use HuggingFace, you need to authenticate:

    ```bash
    huggingface-cli login
    ```
    This will prompt you to enter your Hugging Face credentials.
    <br>

## 3. Usage

The following command will reproduce our results. 


```bash
python main.py -s 300000 -b 64 -lr 5e-4 -ga 8 -n replicateresults -l es tr -vs 10000 20000 30000 40000 -t BPE WordPiece Unigram -m traintoken train eval
```
<br>
Here are the available command-line parameters:

- `-m`: Run mode. Choices: `traintoken`, `train`, `eval`.
- `-l`: Language codes for all datasets. Choices: `es`, `tr`.
- `-t`: Tokenizer types to train. Choices: `BPE`, `WordPiece`, `Unigram`.
- `-vs`: Vocabulary sizes for the tokenizers. Defaults to `[10000]`.
- `-s`: Number of training examples. Defaults to `300000`.
- `-e`: Number of training epochs. Defaults to `1`.
- `-b`: Training batch size. Defaults to `64`.
- `-lr`: Learning rate for training. Defaults to `5e-4`.
- `-ga`: Gradient accumulation for training. Defaults to `8`.
- `-wandb`: Run name for tracking in WandB. Defaults to `tokenizer_run`.

<br>
The training time for all models was 2 days, using a single NVIDIA L40 GPU.




