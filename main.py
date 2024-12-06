"""Main python file to run the pipeline"""

import os
import argparse

import torch
import numpy as np

import torch

from transformers import (
    DistilBertForMaskedLM,
    DistilBertConfig,
    PreTrainedTokenizerFast,
)


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

from src.eval import eval_bpc

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


HUGGINGFACE_TOKEN = "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"

UNK_TOKEN = "[UNK]"
SPL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + [UNK_TOKEN]
MAX_LENGTH = 512

ALL_MODELS_FOLDER = "models"
ALL_TOKENIZERS_FOLDER = "tokenizers"
ALL_RESULTS_FOLDER = "results"


def main(args):
    device, n_gpu = get_available_device()
    print(f"Using device: {device}\nWith {n_gpu} GPUs")

    login(HUGGINGFACE_TOKEN)
    wandb.login()
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    for mode in args.modes:
        for language in args.languages:
            for tokenizer_name in args.tokenizer_types:
                for vocab_size in args.vocab_sizes:
                    for training_size in args.training_sizes:

                        tokenizer_file = os.path.join(
                            ALL_TOKENIZERS_FOLDER,
                            f"tokenizer_{language}_{tokenizer_name}_vs{vocab_size}_ts{training_size}.json",
                        )

                        model_folder = os.path.join(
                            ALL_MODELS_FOLDER,
                            f"model_{language}_{tokenizer_name}_vs{vocab_size}_ts{training_size}",
                        )

                        results_folder = os.path.join(
                            ALL_RESULTS_FOLDER,
                            f"{language}_{tokenizer_name}_vs{vocab_size}_ts{training_size}",
                        )
                        os.makedirs(results_folder, exist_ok=True)

                        dataset = get_oscar_dataset(language, training_size)
                        save_stats_dataset(dataset, results_folder)

                        if mode == "train":
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
                            tokenized_dataset = tokenize_dataset(
                                dataset, tokenizer, MAX_LENGTH
                            )

                            configuration = DistilBertConfig(vocab_size=vocab_size)
                            model = DistilBertForMaskedLM(configuration)

                            max_steps = (
                                int(training_size / args.batch_size / 8) * args.epochs
                            )
                            print(f"Max steps: {max_steps}")
                            trainer = create_mlm_trainer(
                                tokenizer,
                                model,
                                tokenized_dataset,
                                model_folder,
                                args.batch_size,
                                args.learning_rate,
                                args.wandb_run_name,
                                args.epochs,
                                max_steps,
                            )
                            torch.cuda.empty_cache()
                            trainer.train()
                            save_num_params(model, results_folder)

                        elif mode == "eval":
                            print("=" * 50)
                            print(f"Start Evaluation with configuration:")
                            print(f"Language: {language}")
                            print(f"Tokenizer Type: {tokenizer_name}")
                            print(f"Vocabulary Size: {vocab_size}")
                            print(f"Training Size: {training_size}")
                            print("=" * 50)

                            checkpoints = [
                                d
                                for d in os.listdir(model_folder)
                                if d.startswith("checkpoint-")
                            ]
                            print(
                                f"There are {len(checkpoints)} checkpoints for model {model_folder}"
                            )
                            checkpoint_dir = os.path.join(
                                model_folder, checkpoints[0]
                            )  # using the first checkpoint

                            model = DistilBertForMaskedLM.from_pretrained(
                                checkpoint_dir
                            )
                            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                                checkpoint_dir
                            )
                            model.eval()

                            bpc = eval_bpc(
                                model, tokenizer, dataset, dataset_size=training_size
                            )

                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    print("Arguments passed to the script:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    main(args)
