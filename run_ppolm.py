import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppo_lm.generation import generate_text_ppolm


def run_ppolm(
    pretrained_model: str,
    prompt: str,
    num_samples: int,
    bag_of_words: str,
    length: int,
    stepsize: float,
    temperature: float,
    top_k: int,
    sample: bool,
    num_iterations: int,
    warmup_steps: int,
    grad_length: int,
    window_length: int,
    decay: bool,
    kl_scale: float,
    loglevel: str = "error",
) -> list[str]:
    logging.basicConfig(level=getattr(logging, loglevel.upper()))    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if device == "cpu" and torch.backends.mps.is_available() else "cpu"

    logging.info("device=%s", device)
    logging.info("bag_of_words=%s", bag_of_words)
    logging.info("sepsize=%f", stepsize)
    logging.info("temperature=%f", temperature)
    logging.info("top_k=%d", top_k)
    logging.info("sample=%s", str(sample))
    logging.info("kl_scale=%f", kl_scale)

    torch.set_default_device(device)

    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    sentences = generate_text_ppolm(
        model=model,
        tokenizer=tokenizer,
        bow_id=bag_of_words,
        prompt=prompt,
        num_samples=num_samples,
        max_tokens=length,
        stepsize=stepsize,
        kl_scale=kl_scale,
        warmup_steps=warmup_steps,
        grad_length=grad_length,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        window_length=window_length if window_length > 0 else None,
        decay=decay,
    )

    return sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Prompt to condition the generation on",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default="military",
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--stepsize", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--kl_scale", type=float, default=0.0)
    parser.add_argument("--loglevel", type=str, default="info", choices=(
        "debug", "info", "warning", "error", "critical",
    ))

    args = parser.parse_args()
    run_ppolm(**vars(args))
