from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForMaskedLM
from datasets import IterableDataset
import torch

from math import exp
from math import log as ln

from .custom_trainer import CustomTrainer, CustomCallback

TOKENIZER_BPE = "BPE"
TOKENIZER_WPC = "WordPiece"
TOKENIZER_UNI = "Unigram"


def add_arguments(parser):
    parser.add_argument(
        "-m",
        "--modes",
        type=str,
        nargs="+",  # Allow multiple languages
        choices=["train", "eval"],  # "es" for Spanish, "tr" for Turkish
        default=["train"],  # Default to Spanish if no language is specified
        help="Whether to train or evaluate a model. Defaults to ['train'].",
    )
    parser.add_argument(
        "-l",
        "--languages",
        type=str,
        nargs="+",  # Allow multiple languages
        choices=["es", "tr"],  # "es" for Spanish, "tr" for Turkish
        default=["es"],  # Default to Spanish if no language is specified
        help="Language codes for the dataset. Provide one or more languages (choices: 'es' for Spanish, 'tr' for Turkish). Defaults to ['es'].",
    )
    parser.add_argument(
        "-t",
        "--tokenizer-types",
        type=str,
        nargs="+",  # Allow multiple tokenizers
        choices=[TOKENIZER_BPE, TOKENIZER_WPC, TOKENIZER_UNI],
        default=[TOKENIZER_BPE],  # Default to BPE if no tokenizer is specified
        help=f"Which tokenizers to train. Provide one or more options. Choices: {TOKENIZER_BPE}, {TOKENIZER_WPC}, {TOKENIZER_UNI}. Defaults to {TOKENIZER_BPE}.",
    )
    parser.add_argument(
        "-vs",
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[10000],
        help="Vocabulary sizes for the trained tokenizers. Provide one or more values, e.g., 1000 2000.",
    )
    parser.add_argument(
        "-ts",
        "--training-size",
        type=int,
        default=10000,
        help="Training size for the tokenizer. Provide a single value, e.g., 1000.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size. Defaults to 16.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "-wandb",
        "--wandb-run-name",
        type=str,
        default="tokenizer_run",
        help="Run name for tracking WandB.",
    )


def create_mlm_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: AutoModelForMaskedLM,
    tokenized_dataset: IterableDataset,
    model_file: str,
    batch_size: int,
    learning_rate: float,
    run_name: str,
    train_epochs: int,
    max_steps: int,
):
    """
    Creates and configures a Trainer for masked language model training. Masks 15% of the tokens for MLM training.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer used to process the data.
        model (AutoModelForMaskedLM): The model to train.
        tokenized_dataset (IterableDataset): The tokenized dataset for training.
        model_file (str): The file path to save the model.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for training.
        run_name (str): The name for the training run (for tracking).
        train_epochs (int): The number of training epochs.
        max_steps (int): max steps taken. Needs to be set when using IteretableDataset.

    Returns:
        Trainer: The configured Trainer instance.
    """

    def compute_bpc(pred: tuple[torch.tensor, torch.tensor]) -> dict[str, float]:
        """
        Calculates Perplexity (PPL) and Bits Per Character (BPC) for a batch.

        Args:
            pred (Tuple[torch.Tensor, torch.Tensor]):
                    - logits (torch.Tensor): Model output logits of shape [batch_size, seq_len, vocab_size].
                    - labels (torch.Tensor): Ground-truth token labels of shape [batch_size, seq_len].
            tokenizer (PreTrainedTokenizerBase):
                A tokenizer instance used to compute the character-level length.

        Returns:
            Dict[str, float]:
                - "ppl" (float): Perplexity of the batch.
                - "bpc" (float): Bits per character of the batch.
        """
        nonlocal tokenizer
        logits, labels = pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)

        print(logits.shape)
        print(labels.shape)

        mask = (
            labels != -100
        )  # -100 is the default ignore index for padding in Hugging Face#
        labels = labels[mask]
        logits = logits[mask]

        print(labels)
        print(logits)
        print(logits.shape)
        print(labels.shape)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        print(probs.shape)
        true_probs = probs[torch.arange(len(labels)), labels]

        nll = -torch.log(true_probs).mean().item()
        perplexity = exp(nll)
        print(nll)
        print(perplexity)

        total_chars = sum(
            len(tokenizer.decode([label])) for label in labels
        )  # going from token id to token and then to chars
        bpc = ln(perplexity) / ln(2) * (len(labels) / total_chars)

        return {"ppl": float(perplexity), "bpc": float(bpc)}

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=f"./{model_file}",
        overwrite_output_dir=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.01,
        logging_dir="./logs",
        save_strategy="steps",
        logging_steps=1,
        use_cpu=False,
        fp16=True,
        report_to="wandb",
        # gradient_accumulation_steps=8,
        run_name=run_name,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        max_steps=max_steps,
        # evaluation_strategy="steps",
        # per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_bpc,
    )

    # trainer.add_callback(CustomCallback(trainer))
    return trainer
