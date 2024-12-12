from transformers import PreTrainedModel, PreTrainedTokenizerFast
from datasets import Dataset
import torch
from tqdm import tqdm

from math import exp, log
import json
import os

def eval_bpc_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    dataset: Dataset,
    dataset_size: int,
    model_results_folder: str,
) -> float:

    nll_losses = []
    total_tokens = 0
    total_characters = 0

    for example in tqdm(
        dataset, total=dataset_size, desc="Computing BPC", unit="example"
    ):

        n_chars = len(example["text"])
        total_characters += n_chars

        inputs = tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=512
        )  # This is a 1 x n_tokens batch.
        inputs = {
            key: value for key, value in inputs.items() if key != "token_type_ids"
        }

        n_tokens = inputs["input_ids"].size(1)
        total_tokens += n_tokens

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            nll_loss = outputs.loss.item()
            nll_losses.append(nll_loss)

    # Calculate average NLL and perplexity
    averaged_nll = sum(nll_losses) / len(nll_losses)
    perplexity = exp(averaged_nll)
    bpc = (total_tokens / total_characters) * (log(perplexity) / log(2))

    results = {
        "bpc": bpc,
        "perplexity": perplexity,
        "averaged_nll": averaged_nll,
        "total tokens": total_tokens,
        "total characters": total_characters,
        "dataset size": dataset_size,
    }

    results_file = os.path.join(
        model_results_folder,
        f"bpc_eval.json",
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return bpc

def calculate_eval_metrics(
    tokenizer: PreTrainedTokenizerFast,
    dataset: Dataset,
    model_results_folder: str,
    ) -> None:

    num_chars = []
    num_words = []
    num_tokens = []
    
   for example in dataset:
        # number of charcters excluding whitespace
        n_chars = len(''.join(example['text'].split()))
        # number of words (*count periods/commas/colons as one word*)
        n_words = len(example['text'].replace(".", " . ").replace(",", " , ").replace(":", " : ").split())

        # Calculate number of tokens
        inputs = tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=512
        )  
        n_tokens = inputs['input_ids'].size(1)

        num_chars.append(n_chars)
        num_words.append(n_words)
        num_tokens.append(n_tokens)

    # calculate LPT (Length per Tokens)
    lpt = [n_chars/n_tokens for n_chars, n_tokens in zip(num_chars, num_tokens)]

    # calculate fertility
    fertility = [n_tokens/n_words for n_tokens, n_words in zip(num_tokens, num_words)]

    results = {
        "num_chars": num_chars,
        "num_words": num_words,
        "num_tokens": num_tokens,
        "lpt": lpt,
        "fertility": fertility,
    }

    results_file = os.path.join(
        model_results_folder,
        "eval_metrics.json",
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

def calculate_parity(
    languages: list,
    tokenizer_types: list,
    vocab_sizes: list
    ) -> None:
    for tokenizer_name in tokenizer_types:
        for vocab_size in vocab_sizes:
            num_tokens = []
            # Load the results from calculate_eval_metrics
            for language in languages:
                model_results_folder = f"results/{language}_{tokenizer_name}_vs{vocab_size}"
                model_results_path = os.path.join(
                    model_results_folder,
                    'eval_metrics.json'
                    )
                with open(model_results_path) as f:
                    results = json.load(f)
                num_tokens.append(results["num_tokens"])

            parity = [num_tokens1/num_tokens2 for num_tokens1, num_tokens2 in zip(num_tokens[0], num_tokens[1])]

            results = {
                "parity": parity,
            }

            results_path = os.path.join(
                    model_results_folder,
                    'parity.json'
                    )

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)







    
