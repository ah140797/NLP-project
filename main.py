"""Main python file to run the pipeline"""

import os
import argparse

import torch
import numpy as np

import torch

from transformers import DistilBertForMaskedLM, DistilBertConfig


from huggingface_hub import login
import wandb

from src.utils import (
    get_oscar_dataset,
    save_stats_dataset,
    save_num_params,
    get_available_device,
)

from src.tokenization import (
    train_tokenizer,
    tokenize_dataset,
)

from src.training import add_arguments, create_mlm_trainer

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


HUGGINGFACE_TOKEN = "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"

UNK_TOKEN = "[UNK]"
SPL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + [UNK_TOKEN]
MAX_LENGTH = 512


def main(args):
    torch.cuda.empty_cache()
    device, n_gpu = get_available_device()
    print(f"Using device: {device}\nWith {n_gpu} GPUs")

    login(HUGGINGFACE_TOKEN)
    wandb.login()
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    model_folder = "models"
    tokenizer_folder = "tokenizers"
    results_folder = "results"

    for language in args.languages:
        for tokenizer_name in args.tokenizer_types:
            for vocab_size in args.vocab_sizes:
                for training_size in args.training_sizes:

                    tokenizer_file = os.path.join(
                        tokenizer_folder,
                        f"tokenizer_{language}_{tokenizer_name}_{vocab_size}_{training_size}.json",
                    )

                    model_file = os.path.join(
                        model_folder,
                        f"model_{language}_{tokenizer_name}_{vocab_size}_{training_size}",
                    )

                    results_file = os.path.join(
                        results_folder,
                        f"result_{language}_{tokenizer_name}_{vocab_size}_{training_size}.json",
                    )

                    dataset = get_oscar_dataset(language, training_size)
                    save_stats_dataset(dataset, results_file)

                    print("=" * 50)
                    print(f"Start Training with configuration:")
                    print(f"Language: {language}")
                    print(f"Tokenizer Type: {tokenizer_name}")
                    print(f"Vocabulary Size: {vocab_size}")
                    print(f"Training Size: {training_size}")
                    print("=" * 50)

                    tokenizer = train_tokenizer(
                        dataset,
                        tokenizer_name,
                        vocab_size,
                        UNK_TOKEN,
                        SPL_TOKENS,
                        tokenizer_file,
                    )
                    tokenized_dataset = tokenize_dataset(dataset, tokenizer, MAX_LENGTH)

                    configuration = DistilBertConfig(vocab_size=vocab_size)
                    model = DistilBertForMaskedLM(configuration)

                    max_steps = int(training_size / args.batch_size)
                    trainer = create_mlm_trainer(
                        tokenizer,
                        model,
                        tokenized_dataset,
                        model_file,
                        args.batch_size,
                        args.learning_rate,
                        args.wandb_run_name,
                        args.epochs,
                        max_steps,
                    )

                    trainer.train()
                    save_num_params(model, results_file)

                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    print("Arguments passed to the script:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    main(args)
