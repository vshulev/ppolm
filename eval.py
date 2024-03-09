from argparse import ArgumentParser
from collections import defaultdict
import csv
import os
import string
from typing import Tuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from ppo_lm.bow import download_bow


EOT_TOKEN = "<|endoftext|>"


def eval_score(sentences: list[str], topic: str) -> int:
    word_counts: list[int] = []
    bow_path = download_bow(topic)
    with open(bow_path, "r") as file:
        words = file.readlines()
        words = [word.strip().lower() for word in words]

    for sentence in sentences:
        tokens = [tok.strip(string.punctuation)
                  for tok in sentence.strip().lower().split()]
        word_counts.append(sum([tokens.count(word) for word in words]))

    return sum(word_counts)


def eval_success(sentences: list[str], topic: str) -> int:
    tw_dir = os.path.join("data", "test_wordlists")

    # num matches of distinct words
    words = []
    with open(os.path.join(tw_dir, topic + ".txt"), "r") as rf:
        for line in rf:
            words.append(line.strip().lower())

    num_match = 0
    for sent in sentences:
        sent_match = 0
        sent = sent.strip().lower().split()
        sent = [tok.strip(string.punctuation) for tok in sent]
        for word in words:
            if word in sent:
                sent_match += 1
        num_match += sent_match

    return num_match


def eval_perplexity(
    sentences: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str = "cuda"
) -> Tuple[float, float]:
    with torch.no_grad():
        ppl: list[float] = []
        sos_token = tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences)):
            full_tensor_input = tokenizer.encode(
                sos_token + sentence.replace(EOT_TOKEN, " ").strip(),
                return_tensors="pt"
            ).to(device)
            full_loss = model(
                full_tensor_input, labels=full_tensor_input
            )[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl), np.std(ppl)


def eval_grammaticality(
    sentences: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str="cuda"
) -> float:
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(
                model(
                    tokenizer.encode(sent, return_tensors="pt").to(device)
                )[0].flatten(),
                dim=0
            )[1]
            total_good += good_prob
        return (total_good / len(sentences)).cpu().item()


def eval_distinctness(results: dict[str, list[str]]) -> Tuple[float, float, float]:
    d1, d2, d3 = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    total_words = defaultdict(lambda: 0)
    for cw, outputs in results.items():
        for o in outputs:
            o = o.replace(EOT_TOKEN, " ").strip().split(" ")
            o = [str(x) for x in o]
            total_words[cw] += len(o)
            d1[cw].update(o)
            for i in range(len(o) - 1):
                d2[cw].add(o[i] + " " + o[i+1])
            for i in range(len(o) - 2):
                d3[cw].add(o[i] + " " + o[i+1] + " " + o[i+2])
    return_info = []
    avg_d1, avg_d2, avg_d3 = 0, 0, 0
    for cw in total_words.keys():
        return_info.append((cw, "DISTINCTNESS", len(d1[cw]) / total_words[cw], len(d2[cw]) / total_words[cw], len(d3[cw]) / total_words[cw]))
        avg_d1 += len(d1[cw]) / total_words[cw]
        avg_d2 += len(d2[cw]) / total_words[cw]
        avg_d3 += len(d3[cw]) / total_words[cw]
    avg_d1, avg_d2, avg_d3 = avg_d1 / len(total_words.keys()), avg_d2 / len(total_words.keys()), avg_d3 / len(total_words.keys())

    return avg_d1, avg_d2, avg_d3
    # return return_info, (avg_d1, avg_d2, avg_d3)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="CSV file with results")
    parser.add_argument("--batch_size", type=int, default=8, help="max samples at a time")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if device == "cpu" and torch.backends.mps.is_available() else "cpu"
    torch.set_default_device(device)

    sentences_by_topic: dict[str, list[str]] = defaultdict(lambda: [])

    with open(args.results, "r") as rf:
        data = list(csv.DictReader(rf))
        for line in data:
            sentences_by_topic[line["topic"]].append(line["text"])

    all_sentences = [sentence
                     for sentences in sentences_by_topic.values()
                     for sentence in sentences]

    results_by_topic = {topic: {"score": 0.0, "success": 0.0}
                        for topic in sentences_by_topic.keys()}
    results = {"perplexity": 0.0, "grammaticality": 0.0, "distinctness": 0.0}

    for topic, sentences in sentences_by_topic.items():
        score = eval_score(sentences, topic)
        success = eval_success(sentences, topic)

        results_by_topic[topic]["score"] = score
        results_by_topic[topic]["success"] = success

    results["distinctness"] = eval_distinctness(sentences_by_topic)

    grammar_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    grammar_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to(device)
    grammar_model.eval()

    results["grammaticality"] = eval_grammaticality(
        all_sentences, grammar_tokenizer, grammar_model, device=device
    )

    eval_tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
    eval_model = AutoModelForCausalLM.from_pretrained("openai-gpt").to(device)
    eval_model.eval()

    results["perplexity"] = eval_perplexity(
        all_sentences, eval_tokenizer, eval_model, device=device
    )

    print("")

    print("Score:")
    print("====================")
    for t, r in results_by_topic.items():
        print(f"{t}: {r['score']/len(sentences_by_topic[t]):.2f}")
    print(f"Overall: {sum([results['score'] for results in results_by_topic.values()])/len(all_sentences):.2f}")
    print("")

    print("Success:")
    print("====================")
    for t, r in results_by_topic.items():
        print(f"{t}: {r['success']/len(sentences_by_topic[t]):.2f}")
    print(f"Overall: {sum([results['success'] for results in results_by_topic.values()])/len(all_sentences):.2f}")
    print("")

    print("Flunecy metrics:")
    print("====================")
    print(f"Perplexity: {results['perplexity'][0]:.2f}±{results['perplexity'][1]:.2f}")
    print(f"Grammaticality: {results['grammaticality']:.2f}")
    print("")

    print("Diversity metrics:")
    print("====================")
    print(f"Dist-1: {results['distinctness'][0]:.2f}")
    print(f"Dist-2: {results['distinctness'][1]:.2f}")
    print(f"Dist-3: {results['distinctness'][2]:.2f}")
