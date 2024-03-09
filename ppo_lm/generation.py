"""
python run_ppolm.py --prompt="In summary" --stepsize=0.5 --length=80 --sample --window_length=5 -B=military 
"""


import logging
from typing import Optional

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import trange

from ppo_lm.bow import BowId, load_bow
from ppo_lm.objective import perturb_past


BIG_CONST = 1e10


def generate_text_ppolm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    bow_id: BowId,
    prompt: str = "",
    num_samples: int = 1,
    max_tokens: int = 80,
    stepsize: float = 0.2,
    kl_scale: float = 0.0,
    warmup_steps: int = 1,
    grad_length: int = 10000,
    temperature: float = 1.0,
    top_k: int = 0,  # 0 means no top K filtering is performed
    sample: bool = True,
    num_iterations: int = 3,
    window_length: Optional[int] = None,
    decay: bool = False,
) -> list[str]:
    model.eval()

    one_hot_bow = load_bow(bow_id, tokenizer)
    sentences: list[str] = []

    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(tokenizer.bos_token + prompt)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)

    while len(prompt_tokens.shape) < 2:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    for _ in trange(num_samples):
        # Start with the prompt tokens. Value will get updated after each step in the nested loop
        output_so_far = prompt_tokens

        # Obtain the last token from the contet and compute the initial past key-values
        last_token = output_so_far[:, -1:]
        past = model(output_so_far[:, :-1]).past_key_values

        # Generate text autoregressively
        for step in trange(max_tokens):
            # Start with a forward pass to obtain values from the unperturbed model
            with torch.no_grad():
                unpert_outputs = model(output_so_far)
            unpert_logits = unpert_outputs.logits

            # Check if we are above grad max length
            # If yes set step size to 0 (effectively no perturbation)
            if step < warmup_steps or step >= grad_length:
                current_stepsize = 0
            else:
                current_stepsize = stepsize

            pert_past = perturb_past(
                model,
                last_token,
                past,
                one_hot_bow,
                unpert_logits,
                current_stepsize,
                num_iterations,
                window_length,
                decay,
                kl_scale,
            )

            # Generate model outputs using the perturbed past
            with torch.no_grad():
                pert_outputs = model(last_token, past_key_values=pert_past)
            pert_logits = pert_outputs.logits[:, -1, :] / temperature
            pert_logits = top_k_filter(pert_logits, k=top_k)
            pert_probs = torch.nn.functional.softmax(pert_logits, dim=-1)

            # sample or greedy
            if sample:
                last_token = torch.multinomial(pert_probs, num_samples=1)
            else:
                _, last_token = torch.topk(pert_probs, k=1, dim=-1)

            # update context/output_so_far appending the new token
            output_so_far = torch.cat((output_so_far, last_token), dim=1)
            past = pert_outputs.past_key_values

            logging.info(tokenizer.decode(output_so_far.tolist()[0]))

        sentences.append(tokenizer.decode(output_so_far.tolist()[0]))

    return sentences


def top_k_filter(logits: torch.Tensor, k: int):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)