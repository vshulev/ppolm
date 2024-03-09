import logging
from operator import add
from typing import Optional

import torch
import numpy as np
from transformers import PreTrainedModel


SMALL_CONST = 1e-15


def to_var(
    x: torch.Tensor, requires_grad: bool=False, volatile: bool=False
) -> torch.autograd.Variable:
    # TODO this function converts the Tensor x to a torch variable,
    # to enable autograd on the Tensor. Variable is deprecated, instead use
    # a regular torch Tensor with requires_grad=True
    return torch.autograd.Variable(x, requires_grad=requires_grad, volatile=volatile)


def perturb_past(
    model: PreTrainedModel,
    last: torch.Tensor,
    past: torch.Tensor,
    one_hot_bow: torch.Tensor,
    unpert_logits: torch.Tensor,
    stepsize=0.01,
    num_iterations=3,
    window_length: Optional[int]=None,
    decay: bool=False,
    kl_scale: float=0.0,
):
    # Originally past is a list of tuples of length 2, instead stack elements
    # in the tuple to produce a list of tensors.
    past = [torch.stack(attn_block) for attn_block in past]

    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    _, _, _, curr_length, _ = past[0].shape

    if window_length is not None and curr_length > window_length:
        if decay:
            decay_mask = torch.arange(
                0.,
                1.0 + SMALL_CONST,
                1.0 / (window_length)
            )[1:]
        else:
            decay_mask = 1.0

        ones_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([window_length])
            + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([curr_length - window_length])
            + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        )
    else:
        window_mask = torch.ones_like(past[0])

    for iter in range(num_iterations):
        logging.info("iteration=%d", iter + 1)

        curr_perturbation = [
            to_var(torch.from_numpy(p_).to(model.device), requires_grad=True)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape

        all_outputs = model(last, past_key_values=perturbed_past)
        all_logits = all_outputs.logits
        logits = all_logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Compute the Objective

        # Get the unnormalized rewards for each action
        bow_sum = torch.mm(probs, torch.t(one_hot_bow)).sum()
        rewards = torch.t(one_hot_bow).sum(dim=1) / bow_sum

        # Compute the KL ratios
        unpert_probs = torch.nn.functional.softmax(unpert_logits[:, -1, :], dim=-1)
        unpert_probs = (
            unpert_probs + SMALL_CONST *
            (unpert_probs <= SMALL_CONST).float().detach()
        )
        correction = SMALL_CONST * (probs <= SMALL_CONST).float().detach()
        corrected_probs = probs + correction.detach()

        ratios = corrected_probs / unpert_probs
        log_ratios = ratios.log()

        # Compute the KL divergence
        kl = (corrected_probs * log_ratios).sum()
        logging.info("kl=%f", kl.cpu().item())

        # Compute normalized rewards (according to Ziegler et al., 2019)
        norm_rewards = rewards   # - kl_scale * log_ratios

        # PPO clip objective
        epsilon = 0.2
        objective = (
            probs *
            torch.minimum(
                ratios * norm_rewards,
                torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * norm_rewards,
            )
        ).sum() - kl * kl_scale

        logging.info("loss=%s", str(objective.cpu().item()))

        # compute gradients
        objective.backward()

        # normalize gradients
        # TODO use an optimizer such as Adam for variable step size!
        grad = [
            stepsize * (p_.grad * window_mask).data.cpu().numpy()
            for p_ in curr_perturbation
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_).to(model.device), requires_grad=True)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    pert_past = [(attn_block[0], attn_block[1]) for attn_block in pert_past]

    return pert_past
