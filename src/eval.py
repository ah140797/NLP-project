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
