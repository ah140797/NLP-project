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
    total_tokens = 0
    total_characters = 0
    perplexities = []  # match

    for example in tqdm(
        dataset, total=dataset_size, desc="Processing examples", unit="example"
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
            print(f"outputs: {outputs}")
            nll_loss = outputs.loss.item()
            print(f"NLL_LOSS {nll_loss}")
            nll_losses.append(nll_loss)

            # match
            perplexity = exp(nll_loss)

            perplexities.append(perplexity)

    # Calculate average NLL and perplexity
    total_nll_loss = sum(nll_losses)  # Sum of all NLL losses
    averaged_nll = total_nll_loss / total_tokens
    perplexity_ = exp(averaged_nll)

    print(f"Tokens: {total_tokens}")
    print(f"Characters: {total_characters}")
    print(f"Perplexity method 1 {perplexity_}")

    perplexity_match = sum(perplexities) / len(perplexities)
    print(f"Perplexity method 2 {perplexity_match}")

    bpc = (total_tokens / total_characters) * (log(perplexity_) / log(2))
    print(f"BPC {bpc}")

    return bpc
