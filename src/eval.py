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
    #language: str,
    tokenizer: PreTrainedTokenizerFast,
    dataset: IterableDataset,
    model_results_folder: str,
    train_flag: boolean
    ) -> None:

    def get_boundaries(text, tokens):
        boundaries = set()
        current_position = 0
        for token in tokens:
            # Skip whitespace characters
            while current_position < len(text) and text[current_position].isspace():
                current_position += 1
            boundaries.add(current_position)
            current_position += len(token)
        # Add the end position of the last token as a boundary
        boundaries.add(current_position)
        return boundaries

    num_chars = []
    num_words = []
    num_tokens = []
    #f1_scores = []

    #nlp = spacy.load(f"{language}_core_news_sm")
            
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

        # Prepare the inputs to calculate f1
        #morphemes = [token.text for token in nlp(example["text"])]
        #tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
        #bound_morpheme = get_boundaries(example["text"], morphemes)
        #bound_token = get_boundaries(example["text"], tokens)

        #tp = len(bound_token & bound_morpheme)
        #fp = len(bound_token - bound_morpheme)
        #fn = len(bound_morpheme - bound_token)

        #precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        #recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        #f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        #f1_scores.append(f1_score)


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
            #"f1_score": f1_scores
        }

    if train_flag:
        results_file = os.path.join(
            model_results_folder,
            "eval_metrics_train.json",
        )

    else:
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


def calculate_normalized_sequence_length(
    languages: list,
    tokenizer_types: list,
    vocab_sizes: list
    ) -> None:
    for language in languages:
        for vocab in vocab_sizes:
            num_tokens = []
            # Load the results from calculate_eval_metrics
            for tokenizer in tokenizer_types:
                model_results_folder = f"results/{language}_{tokenizer_name}_vs{vocab_size}"
                model_results_path = os.path.join(
                    model_results_folder,
                    'eval_metrics_train.json'
                    )
                with open(model_results_path) as f:
                    results = json.load(f)
                num_tokens.append(results["num_tokens"])

            NSL = []
            for i in range(3):
                for j in range(3):
                    if i < j:
                        nsl = [num_tokens1/num_tokens2 for num_tokens1, num_tokens2 in zip(num_tokens[i], num_tokens[j])]
                        NSL.append(nsl)

            results = {
                "NSL": NSL,
                "tokenizer_names": tokenizer_types
            }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)


def calculate_productivity(
    language: str,
    tokenizer_name: str,
    vocab_size: str,
    tokenizer: PreTrainedTokenizerFast,
    dataset: IterableDataset,
    ) -> None:
    
    productivity_dict = {}
    
    # Load the results from train_tokenizer
    tokenizer_results_path = f"tokenizers/tokenizer_{language}_{tokenizer_name}_vs{vocab_size}.json"
    with open(model_results_path) as f:
        results = json.load(f)

    nlp = spacy.load(f"{language}_core_news_sm")

    for example in tqdm(
        dataset, total=len(dataset), desc="Computing productivity", unit="example"
    ):
        # Get unique words in each text
        words = [token.text for token in nlp(example["text"])]
        for word in words:
            tokens = tokenizer.encode(word)
            for token in tokens:
                if token in productivity_dict:
                    productivity_dict[token] += 1
                else:
                    productivity_dict[token] = 1

    results_path = f"results/{language}_{tokenizer_name}_vs{vocab_size}/productivity.json"

    with open(results_path, "w") as f:
                json.dump(productivity_dict, f, indent=4)






    
