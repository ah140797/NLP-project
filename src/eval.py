from transformers import PreTrainedModel, PreTrainedTokenizerFast
from datasets import Dataset
import torch
from tqdm import tqdm

from math import exp, log


def eval_bpc(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    dataset: Dataset,
    dataset_size: int = None,
) -> tuple[float, int]:

    nll_losses = []
    total_tokens = []
    total_characters = []

    for example in tqdm(
        dataset, total=dataset_size, desc="Processing examples", unit="example"
    ):

        n_chars = len(example["text"])
        total_characters += n_chars

        inputs = tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=512
        )  # This is a 1 x n_tokens batch.
        n_tokens = inputs.input_ids.size(1)
        total_tokens += n_tokens

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            nll_loss = outputs.loss.item()
            nll_losses.append(nll_loss)

    averaged_nll = (torch.cat(nll_losses).sum() / total_tokens).item()
    perplexity = exp(averaged_nll)

    print(f"Tokens: {total_tokens}")
    print(f"Characters: {total_characters}")

    bpc = (total_tokens / total_characters) * (log(perplexity) / log(2))
    print(f"BPC {bpc}")

    return bpc
