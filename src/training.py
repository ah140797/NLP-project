from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    trainer_callback,
)
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForMaskedLM
from datasets import IterableDataset
import torch

from math import exp
from math import log as ln


import torch
from torch.nn.functional import cross_entropy
from copy import deepcopy
from math import log as ln


TOKENIZER_BPE = "BPE"
TOKENIZER_WPC = "WordPiece"
TOKENIZER_UNI = "Unigram"


def add_arguments(parser):
    parser.add_argument(
        "-m",
        "--modes",
        type=str,
        nargs="+",
        choices=["train", "eval"],
        default=["train"],
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
        "-s",
        "--dataset-size",
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
        "-ga",
        "--gradient-accumulation",
        type=int,
        default=2,
        help="Gradient accumulation for training.",
    )
    parser.add_argument(
        "-n",
        "--wandb-run-name",
        type=str,
        default="tokenizer_run",
        help="Run name for tracking WandB.",
    )


class CustomCallback(trainer_callback.TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if control.should_log:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="model"
            )
            return control_copy


def create_mlm_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: AutoModelForMaskedLM,
    tokenized_dataset: IterableDataset,
    model_file: str,
    batch_size: int,
    learning_rate: float,
    run_name: str,
    train_epochs: int,
    gradient_accumulation: int,
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

    def compute_metrics(eval_pred):
        """Computes the perplexity metric for language modeling.

        Args:
            eval_pred (EvalPrediction): The evaluation prediction object from the Trainer.

        Returns:
            dict: A dictionary containing the perplexity metric.
        """
        nonlocal tokenizer

        logits = torch.tensor(eval_pred.predictions)
        labels = torch.tensor(eval_pred.label_ids)

        # Convert logits to probabilities
        mask = (
            labels != -100
        )  # -100 is the default ignore index for padding in Hugging Face#

        labels = labels[mask]
        logits = logits[mask]

        probs = cross_entropy(logits, labels)

        # Calculate perplexity
        perplexity = torch.exp(probs)

        total_no_tokens = labels.size(0)

        total_chars = sum(len(tokenizer.decode([label])) for label in labels)

        bpc = ln(perplexity) / ln(2) * (total_no_tokens / total_chars)
        return {"perplexity": perplexity.item(), "bpc": bpc}

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=f"./{model_file}",
        overwrite_output_dir=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        logging_dir="./logs",
        save_strategy="steps",
        logging_steps=2,
        use_cpu=False,
        fp16=True,
        report_to="wandb",
        run_name=run_name,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        gradient_accumulation_steps=gradient_accumulation,
        max_steps=max_steps,
        # evaluation_strategy="steps",
        # per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(CustomCallback(trainer))
    return trainer
