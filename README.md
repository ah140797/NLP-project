# Welcome to the Shinkansen NLP Project


### Requirements

1. **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Login to WandB for experiment tracking:**

    If you're using `wandb` for experiment tracking, you need to authenticate using your WandB account. Run the following command:

    ```bash
    wandb login
    ```

    This will prompt you to enter an API key from [wandb](https://wandb.ai/).

## Available Parameters

- `--languages`: Language codes for the dataset (e.g., `es`, `tr`). Defaults to `['es']`.
- `--tokenizer-types`: Tokenizer types to train. Choices: `BPE`, `WordPiece`, `Unigram`. Defaults to `['BPE']`.
- `--vocab-sizes`: Vocabulary sizes for the tokenizers. Defaults to `[10000]`.
- `--training-sizes`: Number of training examples. Defaults to `[10000]`.
- `--epochs`: Number of training epochs. Defaults to `1`.
- `--batch-size`: Training batch size. Defaults to `264`.
- `--learning-rate`: Learning rate for training. Defaults to `5e-5`.
- `--wandb-run-name`: Run name for tracking in WandB. Defaults to `tokenizer_run`.

## Example Usage

```bash
python main.py --languages es tr --tokenizer-types BPE WordPiece --vocab-sizes 10000 20000 --training-sizes 10000 20000 --epochs 3 --batch-size 16 --learning-rate 5e-5 --wandb-run-name "example_run"
```


