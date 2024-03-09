import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm
import yaml

from run_ppolm import run_ppolm


BOW_TOPICS = [
    "science",
    "space",
    "politics",
    "military",
    "religion",
    "technology",
    "legal",
]


def generate_sentences(model: str, num_samples: int, length: int, topics: str):
    with open("hyperparameters.yaml", "r") as hyperparams_f:
        config = yaml.safe_load(hyperparams_f)

    with open("data/contexts.txt", "r") as file:
        contexts = file.readlines()
        contexts = [context.strip() for context in contexts]

    all_dataframes: list[pd.DataFrame] = []

    for context in tqdm(contexts):
        for bow_topic in topics.split(","):
            logging.info("context=%s,topic=%s", context, bow_topic)

            sentences = run_ppolm(
                pretrained_model=model,
                prompt=context,
                num_samples=num_samples,
                bag_of_words=bow_topic,
                length=length,
                **{**config["default"], **config.get(bow_topic, {})},
            )

            results_path = f"data/results/checkpoints/{context}_{bow_topic}.csv"
            df = pd.DataFrame(
                [
                    [bow_topic, context, sentence]
                    for sentence in sentences
                ],
                columns=["topic", "prompt", "text"],
            )

            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            df.to_csv(results_path, index=False)

            all_dataframes.append(df)

    results_path = "data/results/ppolm.csv"
    df = pd.concat(all_dataframes, ignore_index=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--topics", type=str, default=",".join(BOW_TOPICS))

    args = parser.parse_args()
    generate_sentences(**vars(args))
