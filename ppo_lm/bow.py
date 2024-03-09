import os
from typing import Literal

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer


BowId = Literal[
    "legal", "military", "monsters", "politics", "positive_words", "religion",
    "science", "space", "technology",
]


BAG_OF_WORDS_ARCHIVE_MAP: dict[BowId, str] = {
    "legal": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    "military": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    "monsters": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    "politics": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    "positive_words": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    "religion": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    "science": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    "space": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    "technology": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}


def load_bow(bow_id: BowId, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """
    Loads the bag of words with the given id and tokenizes it using the given tokenizer.

    Args:
        bow_id (BowId): The id of the bag of words (e.g. "legal", "military", etc.)
        tokenizer (PreTrainedTokenizer): The PreTrainedTokenizer to use.
    
    Returns:
        torch.Tensor: A tensor of dimensions BxV, where B is the number of words
                     in the BoW and V is the vocabulary size.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, add_prefix_space=True)

    tokenized_bow = tokenize_bow(bow_id, tokenizer)

    return build_bows_one_hot_vectors(tokenized_bow, tokenizer)


def download_bow(bow_id: BowId) -> str:
    """
    Downloads the bag of words with the given id.


    Args:
        bow_id (BowId): The id of the bag of words (e.g. "legal", "military", etc.)
    
    Returns:
        str: The path to the downloaded bag of words.
    """
    bow_url = BAG_OF_WORDS_ARCHIVE_MAP[bow_id]
    bow_fname = bow_url.split("/")[-1]
    bow_path = f"data/bow/{bow_fname}"

    # Create dirs if not exist
    os.makedirs(os.path.dirname(bow_path), exist_ok=True)

    torch.hub.download_url_to_file(bow_url, bow_path)

    return bow_path


def tokenize_bow(bow_id: BowId, tokenizer: PreTrainedTokenizer) -> list[list[int]]:
    """
    Tokenizes the words in the bag of words with the given id using the given tokenizer.

    Args:
        bow_id (BowId): The id of the bag of words (e.g. "legal", "military", etc.)
        tokenizer (PreTrainedTokenizer): The PreTrainedTokenizer to use.
    
    Returns:
        list[list[int]]: A list of lists of token ids, one list per each word in the bag of words.
    """

    assert bow_id in BAG_OF_WORDS_ARCHIVE_MAP, f"Invalid BoW id: {bow_id}"

    bow_path = download_bow(bow_id)

    with open(bow_path, "r") as f:
        words = f.read().strip().split("\n")

    return [tokenizer.encode(word.strip())
            for word in words]


def build_bows_one_hot_vectors(
    tokenized_bow: list[list[int]], tokenizer: PreTrainedTokenizer
) -> torch.Tensor:
    """
    Builds a one-hot matrix of dimensions BxV, where B is the number of words
    in the BoW and V is the vocabulary size. Each row is a sparse vector
    representation of one of the words in the BoW. There are vocab. size
    columns, for each row only one column is equal to 1.

    Args:
        tokenized_bow (list[list[int]]): A list of lists of token ids, one list
                                         per each word in the bag of words.
        tokenizer (PreTrainedTokenizer): The PreTrainedTokenizer to use.    

    Returns:
        torch.Tensor: A tensor of dimensions BxV, where B is the number of words
                     in the BoW and V is the vocabulary size.
    """

    # Filter only words which are encoded as a max. 1 token
    tokenized_bow = list(filter(lambda x: len(x) <= 1, tokenized_bow))

    # Convert the tokenized BoW to a tensor
    tokenized_bow_tensor = torch.tensor(tokenized_bow)

    num_words_in_bow = tokenized_bow_tensor.shape[0]

    vocab_size = tokenizer.vocab_size
    # Fixes issue with tokenizer and model having different vocab size.
    # Additional info here: https://huggingface.co/microsoft/phi-2/discussions/22
    if tokenizer.name_or_path in ["microsoft/phi-1_5", "microsoft/phi-2"]:
        vocab_size = 51200

    # Construct a one-hot matrix of dimensions BxV.
    # Each row is a sparse vector representation of one of the words in the BoW
    # There are vocab. size columns, for each row only one column is equal
    # to 1.
    one_hot_bow = torch.zeros(num_words_in_bow, vocab_size)
    one_hot_bow.scatter_(1, tokenized_bow_tensor, 1)

    return one_hot_bow
