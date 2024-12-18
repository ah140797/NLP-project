# Welcome to the Shinkansen NLP Project
Anders Hjulmand, Eisuke Okuda, Andreas Flensted

## 1. Introduction

Investgate agglutinativs vs fusional...

Tinybert was trained with a Masked Language Modeling (MLM) objective by masking $15\%$ of tokens. We trained for $1$ epoch with a learning rate of $5e-4$, batch size of $64$ and gradient accumulation of $8$.

We trained $24$ models, varying by language (Spanish, Turkish), tokenizer (BPE, Wordpiece, Unigram), and vocabulary size ($10$K, $20$K, $30$K, $40$K). 

## 2. Setup

1. **Clone repository**
   ```bash
    git clone https://github.com/eisuke119/Research-Project.git
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

The following command will reproduce our results:


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






