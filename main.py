"""Main python file to run the pipeline"""

import os
import argparse

import torch
from transformers import BertConfig, AutoModelForMaskedLM, PreTrainedTokenizerFast

from huggingface_hub import login
import wandb

from src.utils import (
    get_oscar_dataset,
    preprocess_dataset,
    save_stats_dataset,
    save_num_params,
    get_available_device,
)

from src.tokenization import train_tokenizer, tokenize_dataset

from src.training import add_arguments, create_mlm_trainer

from src.eval import eval_bpc

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


HUGGINGFACE_TOKEN = "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"
TINYBERT_CONFIG = "huawei-noah/TinyBERT_General_4L_312D"

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

    training_size = args.training_size

    for mode in args.modes:
        for language in args.languages:
            dataset_results_folder = os.path.join(
                ALL_RESULTS_FOLDER,
                f"dataset_stats {language}",
            )
            os.makedirs(dataset_results_folder, exist_ok=True)

            dataset = get_oscar_dataset(language, training_size)

            processed_dataset, processed_training_size = preprocess_dataset(
                dataset, language
            )
            
            if mode == 'eval':
                ...
                continue#
                

            save_stats_dataset(dataset, dataset_results_folder, language)

            for tokenizer_name in args.tokenizer_types:
                for vocab_size in args.vocab_sizes:

                    tokenizer_file = os.path.join(
                        ALL_TOKENIZERS_FOLDER,
                        f"tokenizer_{language}_{tokenizer_name}_vs{vocab_size}_ts{processed_training_size}.json",
                    )

                    model_folder = os.path.join(
                        ALL_MODELS_FOLDER,
                        f"model_{language}_{tokenizer_name}_vs{vocab_size}_ts{processed_training_size}",
                    )

                    model_results_folder = os.path.join(
                        ALL_RESULTS_FOLDER,
                        f"{language}_{tokenizer_name}_vs{vocab_size}_ts{processed_training_size}",
                    )

                    os.makedirs(model_results_folder, exist_ok=True)

                    if mode == "train":
                        print("=" * 50)
                        print(f"Start Training with configuration:")
                        print(f"Language: {language}")
                        print(f"Tokenizer Type: {tokenizer_name}")
                        print(f"Vocabulary Size: {vocab_size}")
                        print(f"Training Size (Processed): {processed_training_size}")
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
                
                config = BertConfig.from_pretrained(
                        TINYBERT_CONFIG, vocab_size=vocab_size
                    )
                model = AutoModelForMaskedLM.from_config(config)

                max_steps = (
                    int(processed_training_size / args.batch_size / 8) * args.epochs
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
                save_num_params(model, model_results_folder)

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
                    f"There are {len(checkpoints)} checkpoints for {model_folder}"
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
