from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import (
    BpeTrainer,
    WordPieceTrainer,
    UnigramTrainer,
)
from transformers import PreTrainedTokenizerFast
from tokenizers.normalizers import (
    Sequence,
    NFC,
    StripAccents,
    Sequence as NormalizerSequence,
)
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence as PreSequence
from tokenizers.processors import TemplateProcessing

from datasets import IterableDataset

from src.utils import dataset_text_iterator


TOKENIZER_BPE = "BPE"
TOKENIZER_WPC = "WordPiece"
TOKENIZER_UNI = "Unigram"


def prepare_tokenizer_trainer(
    alg: str, vocabulary_size: int, unk_token: str, spl_tokens: list[str]
) -> tuple[Tokenizer, object]:
    """
    Prepares a tokenizer and its trainer based on the selected algorithm. Also NFC and strip-accent normalizes.

    Args:
        alg (str): The tokenizer algorithm to use. Options are 'BPE', 'WPC', or 'UNI'.
        vocabulary_size (int): The desired size of the vocabulary.
        unk_token (str): The token to use for unknown words.
        spl_tokens (list[str]): A list of special tokens to include in the tokenizer.

    Returns:
        tuple: A tuple containing:
            - Tokenizer object: A tokenizer instance (BPE, WordPiece, or Unigram).
            - Trainer object: The corresponding trainer object (BpeTrainer, WordPieceTrainer, or UnigramTrainer) based on the algorithm.
    """

    if alg == TOKENIZER_BPE:
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=vocabulary_size)
    elif alg == TOKENIZER_WPC:
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(
            special_tokens=spl_tokens,
            vocab_size=vocabulary_size,
        )
    elif alg == TOKENIZER_UNI:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            unk_token=unk_token, special_tokens=spl_tokens, vocab_size=vocabulary_size
        )
    else:
        exit(
            f"Unknown tokenizer type. Please use either {TOKENIZER_BPE}, {TOKENIZER_WPC}, or {TOKENIZER_UNI}"
        )

    tokenizer.normalizer = NormalizerSequence([NFC(), StripAccents()])
    tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])

    tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <SEP>",
        pair="<CLS> $A <SEP> $B:1 <SEP>:1",
        special_tokens=[("<CLS>", 1), ("<SEP>", 2)],
    )

    return tokenizer, trainer


def train_tokenizer(
    dataset: IterableDataset,
    alg: str,
    vocabulary_size: int,
    unk_token: str,
    spl_tokens: list[str],
    tokenizer_file: str,
) -> Tokenizer:
    """
    Trains a tokenizer on the given dataset and saves it to a file.

    Args:
        dataset (IterableDataset): The dataset to train the tokenizer on.
        alg (str): The tokenizer algorithm ('BPE', 'WordPiece', 'Unigram').
        vocabulary_size (int): The size of the vocabulary.
        unk_token (str): The token used for unknown words.
        spl_tokens (list[str]): Special tokens to add to the tokenizer.
        tokenizer_file (str): The file path to save the trained tokenizer.

    Returns:
        Tokenizer: The trained tokenizer.
    """

    tokenizer, trainer = prepare_tokenizer_trainer(
        alg, vocabulary_size, unk_token, spl_tokens
    )

    tokenizer.train_from_iterator(dataset_text_iterator(dataset), trainer)
    tokenizer.save(tokenizer_file)
    tokenizer = Tokenizer.from_file(tokenizer_file)

    return tokenizer


def tokenize_dataset(
    dataset: IterableDataset, tokenizer: PreTrainedTokenizerFast, max_length: int
) -> IterableDataset:
    """
    Tokenize and truncate the input dataset.

    Args:
        dataset (IterableDataset): The HuggingFace dataset to tokenize.
        tokenizer (Tokenizer): Tokenizer to use for encoding.
        max_length (int): Maximum sequence length.

    Returns:
        IterableDataset: Tokenized dataset.
    """
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        ),
        batched=True,
        remove_columns=["text"],
    )
    return tokenized_dataset


def load_tokenizer(tokenizer_file: str):

    tokenizer = Tokenizer.from_file(tokenizer_file)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<UNK>",
        pad_token="<PAD>",
        cls_token="<CLS>",
        sep_token="<SEP>",
        mask_token="<MASK>",
        return_special_tokens_mask=True,
        return_token_type_ids=False,
    )

    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "cls_token": "<CLS>",
            "sep_token": "<SEP>",
            "mask_token": "<MASK>",
        }
    )

    return tokenizer