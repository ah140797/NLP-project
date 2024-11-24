from datasets import load_dataset
from datasets import IterableDataset
import torch


def get_oscar_dataset(language: str, training_size: int) -> IterableDataset:
    """Loads the OSCAR dataset in streaming-mode (iteratabledataset) for a specified language and training size.

    Args:
        language (str): The language code for the desired language subset of the OSCAR corpus (e.g., 'en' for English, 'tr' for Turkish).
        training_size (int): The number of samples to retrieve from the dataset.

    Returns:
        IterableDataset: An iteretable dataset object of the OSCAR corpus, for language and training size specified.
    """
    dataset = load_dataset(
        "oscar-corpus/oscar",
        language=language,
        streaming=True,
        split="train",  # optional, but the dataset only has a train split
    )

    dataset = dataset.take(training_size)

    return dataset


def dataset_text_iterator(dataset: IterableDataset):
    """Yields the 'text' column from an iterable dataset.


    Args:
        dataset (IterableDataset): An iterable dataset where each is expected to be a dictionary with a 'text' field.

    Yields:
        str: The text content from each sample in the dataset.
    """
    for sample in dataset:
        yield sample["text"]


def count_words_in_dataset(dataset: IterableDataset) -> None:
    """
    Counts and prints the total number of words in the dataset.

    Args:
        dataset (IterableDataset): The dataset to count words in.

    Returns:
        None
    """
    word_count = 0
    for sample in dataset:
        text = sample["text"]  # Assuming each sample has a "text" field
        words = text.split()  # Split the text into words
        word_count += len(words)  # Count the words and accumulate

    print(f"Total words in dataset: {word_count}")


def get_available_device():
    """
    Returns the best available device for PyTorch computations.
    - If CUDA (GPU) is available, it returns 'cuda'.
    - If MPS (Apple GPU) is available, it returns 'mps'.
    - Otherwise, it returns 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        return device, gpu_count
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # returning to correctly handle batch size.
        return device, 1
    else:
        device = torch.device("cpu")
        # returning to correctly handle batch size.
        return device, 1
