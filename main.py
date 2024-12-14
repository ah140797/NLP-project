"""Main python file to run the pipeline"""

import os
import argparse

import torch
from torch import nn
from transformers import BertConfig, AutoModelForMaskedLM, PreTrainedTokenizerFast
from datasets import load_from_disk

from huggingface_hub import login
import wandb

from src.utils import (
    get_oscar_dataset,
    load_flores_dataset,
    load_massive_dataset,
    load_turkish_treebanks_dataset,
    preprocess_dataset,
    save_stats_dataset,
    save_num_params,
    get_available_device,
    load_model_from_checkpoint,
)

from src.tokenization import train_tokenizer, tokenize_dataset, load_tokenizer

from src.training import add_arguments, create_mlm_trainer

from src.eval import eval_bpc_ppl, calculate_eval_metrics, calculate_parity, calculate_productivity


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


HUGGINGFACE_TOKEN = "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"
TINYBERT_CONFIG = "huawei-noah/TinyBERT_General_4L_312D"

# UNK_TOKEN = "[UNK]"
# SPL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + [UNK_TOKEN]

UNK_TOKEN = "<UNK>"
SPL_TOKENS = ["<UNK>", "<CLS>", "<SEP>", "<MASK>", "<PAD>"]

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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"  # remove allocated memory thats not in use
    )
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    dataset_size = args.dataset_size

    for language in args.languages:
        dataset_results_folder = os.path.join(
            ALL_RESULTS_FOLDER,
            f"dataset_stats_{language}",
        )
        os.makedirs(dataset_results_folder, exist_ok=True)

        dataset = get_oscar_dataset(language, dataset_size)

        processed_dataset, processed_dataset_size = preprocess_dataset(
            dataset, language
        )

        save_stats_dataset(processed_dataset, dataset_results_folder, language)

        for mode in args.modes:
            for tokenizer_name in args.tokenizer_types:
                for vocab_size in args.vocab_sizes:

                    tokenizer_file = os.path.join(
                        ALL_TOKENIZERS_FOLDER,
                        f"tokenizer_{language}_{tokenizer_name}_vs{vocab_size}.json",
                    )

                    model_folder = os.path.join(
                        ALL_MODELS_FOLDER,
                        f"model_{language}_{tokenizer_name}_vs{vocab_size}",
                    )

                    model_results_folder = os.path.join(
                        ALL_RESULTS_FOLDER,
                        f"{language}_{tokenizer_name}_vs{vocab_size}",
                    )

                    os.makedirs(model_results_folder, exist_ok=True)

                    if mode == "traintoken":
                        print("=" * 50)
                        print(f"Start Tokenizer Training:")
                        print(f"Language: {language}")
                        print(f"Tokenizer Type: {tokenizer_name}")
                        print(f"Vocabulary Size: {vocab_size}")
                        print(f"Dataset Size (Processed): {processed_dataset_size}")
                        print("=" * 50)

                        tokenizer = train_tokenizer(
                            processed_dataset,
                            tokenizer_name,
                            vocab_size,
                            UNK_TOKEN,
                            SPL_TOKENS,
                            tokenizer_file,
                        )

                    elif mode == "train":
                        print("=" * 50)
                        print(f"Start Training with configuration:")
                        print(f"Language: {language}")
                        print(f"Tokenizer Type: {tokenizer_name}")
                        print(f"Vocabulary Size: {vocab_size}")
                        print(f"Dataset Size (Processed): {processed_dataset_size}")
                        print("=" * 50)

                        tokenizer = load_tokenizer(tokenizer_file)

                        tokenized_dataset = tokenize_dataset(
                            processed_dataset, tokenizer, MAX_LENGTH
                        )

                        import json

                        tokenized_dataset_list = list(tokenized_dataset)
                        # Save as JSON
                        with open("tokenized_dataset.json", "w") as f:
                            json.dump(tokenized_dataset_list, f)
                        break

                        config = BertConfig.from_pretrained(
                            TINYBERT_CONFIG, vocab_size=vocab_size
                        )
                        model = AutoModelForMaskedLM.from_config(config)

                        if n_gpu > 1:
                            model = nn.DistributedDataParallel(model)

                        max_steps = (
                            int(
                                processed_dataset_size
                                / args.batch_size
                                / args.gradient_accumulation
                            )
                            * args.epochs
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
                            args.gradient_accumulation,
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
                        print(f"Dataset Size (Processed): {processed_dataset_size}")
                        print("=" * 50)

                        checkpoint_dir = load_model_from_checkpoint(model_folder)

                        tokenizer = PreTrainedTokenizerFast.from_pretrained(
                            checkpoint_dir
                        )

                        #model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir)

                        #model.eval()

                        #bpc = eval_bpc_ppl(
                        #    model,
                        #    tokenizer,
                        #    processed_dataset,
                        #    processed_dataset_size,
                        #    model_results_folder,
                        #)

                        eval_ds_flores = load_flores_dataset(language)
                        eval_ds_massive = load_massive_dataset(language)

                        calculate_eval_metrics(tokenizer, eval_ds_flores, model_results_folder, "flores")
                        calculate_eval_metrics(tokenizer, eval_ds_massive, model_results_folder, "massive")
                        calculate_productivity(language, tokenizer_name, vocab_size, tokenizer, eval_ds_flores, "flores")
                        calculate_productivity(language, tokenizer_name, vocab_size, tokenizer, eval_ds_massive, "massive")

                        if language == "tr":
                            eval_ds_treebanks = load_turkish_treebanks_dataset()
                            calculate_f1_score(tokenizer, eval_ds_treebanks, model_results_folder)


    if mode == "eval":
        # Needs to be commented out unless arg.tokenizer_types has all 3 types
        #calculate_normalized_sequence_length(args.languages, args.tokenizer_types, args.vocab_sizes, "flores")
        #calculate_normalized_sequence_length(args.languages, args.tokenizer_types, args.vocab_sizes, "massive")
        calculate_parity(args.languages, args.tokenizer_types, args.vocab_sizes, "flores")
        calculate_parity(args.languages, args.tokenizer_types, args.vocab_sizes, "massive")


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    print("Arguments passed to the script:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    main(args)
