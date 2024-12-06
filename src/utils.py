import json
import os
import time

from datasets import load_dataset
from datasets import IterableDataset
import torch
from torch import nn

from fast_langdetect import detect


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
        trust_remote_code=True,
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


def text_normalise(text: str) -> str:
    """Normalizes the text by removing newlines and converting it to lowercase."""
    text = text.replace("\n", "")
    text = text.lower()
    return text


def detect_language(text: str, target_language: str) -> bool:
    """Detects the language of the text and returns whether it matches the target language."""

    try:
        detected_lang = detect(text)["lang"]
        return detected_lang == target_language  # returns true if they are the same

    except Exception as e:
        print(f"Error detecting language: {e}")
        return False


def preprocess_dataset(
    dataset: IterableDataset, target_language: str
) -> tuple[IterableDataset, int]:
    """Preprocess dataset and filter by language."""
    start_time = time.time()

    def generator():
        nonlocal processed_training_size
        for example in dataset_text_iterator(dataset):
            example = text_normalise(example)

            if detect_language(example, target_language):
                processed_training_size += 1
                yield {"text": example}

    processed_training_size = 0
    processed_dataset = IterableDataset.from_generator(generator)
    list(processed_dataset)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Preprocessing time: {processing_time:.2f} seconds")
    print(f"Processed dataset size: {processed_training_size}")

    return processed_dataset, processed_training_size


def save_stats_dataset(
    dataset: IterableDataset, results_folder: str, language: str
) -> None:
    """
    Counts and saves the total number of words in the dataset, the size of the dataset in MB, and the number of examples.

    Args:
        dataset (IterableDataset): The dataset to count words in.
        results_folder (str): path where results are saved

    Returns:
        None
    """
    word_count = 0
    sample_count = 0
    total_size_bytes = 0

    for sample in dataset:
        sample_count += 1
        text = sample["text"]
        words = text.split()
        word_count += len(words)

        # Get the size of the sample in bytes
        total_size_bytes += len(str(sample).encode("utf-8"))

    # Convert bytes to MB
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total words in dataset: {word_count}")
    print(f"Total number of examples in dataset: {sample_count}")
    print(f"Total size of dataset: {total_size_mb:.2f} MB")

    results = {
        "word_count": word_count,
        "sample_count": sample_count,
        "total_size_mb": total_size_mb,
    }

    results_file = os.path.join(
        results_folder,
        f"dataset_stats{language}_.json",
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return


def save_num_params(model: nn.Module, results_folder: str) -> None:
    """
    Print and saves the total number of parameters in the model in millions.

    Args:
        model (nn.Module): The model whose parameters are to be counted.
        results_path (str): path where results are saved
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_params_million = num_params / 1e6
    print(f"Number of parameters in the model: {num_params_million:.2f}M")

    results = {
        "num_params_million": num_params_million,
    }

    results_file = os.path.join(
        results_folder,
        f"model_num_params.json",
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return


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
