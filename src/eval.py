from transformers import PreTrainedTokenizerFast
import torch

from math import exp
from math import log as ln


def compute_bpc(
    pred: tuple[torch.tensor, torch.tensor], tokenizer: PreTrainedTokenizerFast
) -> dict[str, float]:
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

    logits, labels = pred
    mask = (
        labels != -100
    )  # -100 is the default ignore index for padding in Hugging Face
    labels = labels[mask]
    logits = logits[mask]

    print(labels)
    print(logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    true_probs = probs[torch.arange(len(labels)), labels]

    nll = -torch.log(true_probs).mean().item()
    perplexity = exp(nll)

    total_chars = sum(
        len(tokenizer.decode([label])) for label in labels
    )  # going from token id to token and then to chars
    bpc = ln(perplexity) / ln(2) * (len(labels) / total_chars)

    return {"ppl": float(perplexity), "bpc": float(bpc)}
