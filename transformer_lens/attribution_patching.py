"""Attribution Patching.

A module for computing gradient-based approximations of activation patching effects. This
implements "attribution patching" (also known as "AtP" or "gradient-based activation patching"),
which provides a fast, linearized approximation of the causal effect of patching each activation.

Background:

    Standard activation patching (see :mod:`transformer_lens.patching`) requires one forward pass
    per patch site, which is O(N) in the number of components. Attribution patching instead computes
    all effects simultaneously using a single forward pass and a single backward pass, making it
    O(1) regardless of the number of components.

    The core idea is to use a first-order Taylor expansion. The effect of replacing activation
    ``a_corrupted`` with ``a_clean`` is approximated by:

        effect ≈ (a_clean - a_corrupted) · ∂metric/∂a

    where the gradient is evaluated at the corrupted activations. This gives the same directional
    information as full activation patching for most practical cases, and is exact when the model
    is locally linear with respect to the patched activation.

    This technique was introduced in:
    - Neel Nanda, "Attribution Patching: Activation Patching At Industrial Scale" (2023)
    - Syed et al., "Attribution Patching Outperforms Automated Circuit Discovery" (2023)

Usage:

    >>> import torch
    >>> from transformer_lens import HookedTransformer
    >>> from transformer_lens.attribution_patching import attribution_patch_residual_stream
    >>> model = HookedTransformer.from_pretrained("gpt2-small")  # doctest: +SKIP
    >>> clean_tokens = model.to_tokens("The Eiffel Tower is in")  # doctest: +SKIP
    >>> corrupted_tokens = model.to_tokens("The Colosseum is in")  # doctest: +SKIP
    >>> def metric(logits):  # doctest: +SKIP
    ...     # logit difference between "Paris" and "Rome"
    ...     paris_token = model.to_single_token(" Paris")
    ...     rome_token = model.to_single_token(" Rome")
    ...     return logits[0, -1, paris_token] - logits[0, -1, rome_token]
    >>> results = attribution_patch_residual_stream(  # doctest: +SKIP
    ...     model, clean_tokens, corrupted_tokens, metric
    ... )
    >>> results.shape  # [n_layers, pos]  # doctest: +SKIP
    torch.Size([12, 6])
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from typing_extensions import Literal

import transformer_lens.utilities as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer


def _get_cache_with_grads(
    model: HookedTransformer,
    tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
    names_filter: Optional[Callable[[str], bool]] = None,
) -> Tuple[ActivationCache, Dict[str, torch.Tensor]]:
    """Run the model with caching enabled and compute gradients of the metric w.r.t. cached activations.

    Args:
        model: The HookedTransformer model.
        tokens: Input token tensor.
        metric: A function from logits to a scalar metric.
        names_filter: Optional filter for which hook points to cache. If None, caches all.

    Returns:
        A tuple of (activation_cache, grad_dict) where grad_dict maps hook names to their gradients.
    """
    model.reset_hooks()
    cache_dict: Dict[str, torch.Tensor] = {}
    grad_dict: Dict[str, torch.Tensor] = {}

    def save_hook_with_grad(tensor: torch.Tensor, hook) -> None:
        # Store the activation and register a gradient hook
        tensor.retain_grad()
        cache_dict[hook.name] = tensor

    # Determine which hooks to attach
    if names_filter is None:
        names_filter = lambda name: True

    fwd_hooks = [
        (name, save_hook_with_grad)
        for name in model.hook_dict.keys()
        if names_filter(name)
    ]

    # Run forward pass with hooks (don't reset at end so backward works)
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=fwd_hooks,
        reset_hooks_end=True,
    )

    # Compute metric and backpropagate
    metric_value = metric(logits)
    metric_value.backward()

    # Extract gradients
    for name, tensor in cache_dict.items():
        if tensor.grad is not None:
            grad_dict[name] = tensor.grad.detach()
        else:
            grad_dict[name] = torch.zeros_like(tensor)

    # Detach cache tensors
    cache_dict = {k: v.detach() for k, v in cache_dict.items()}

    model.zero_grad()

    return ActivationCache(cache_dict, model), grad_dict


def attribution_patch_residual_stream(
    model: HookedTransformer,
    clean_tokens: Int[torch.Tensor, "batch pos"],
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
    position: Literal["resid_pre", "resid_mid", "resid_post"] = "resid_pre",
) -> Float[torch.Tensor, "n_layers pos"]:
    """Compute attribution patching scores for the residual stream at each layer and position.

    This approximates the effect of patching each residual stream vector from the corrupted run
    to the clean run, using gradients computed on the corrupted run.

    The approximation is: effect[layer, pos] ≈ (clean_act - corrupted_act)[pos] · grad[pos]

    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokens for the clean run (the "correct" input).
        corrupted_tokens: Tokens for the corrupted run (the "incorrect" input).
        metric: A function from logits to a scalar metric. Should return a scalar tensor.
            Higher values should correspond to more "correct" behavior.
        position: Which residual stream position to patch. One of "resid_pre", "resid_mid",
            "resid_post".

    Returns:
        A tensor of shape [n_layers, pos] containing the approximate patching effect at each
        layer and position. Positive values indicate that patching (replacing corrupted with
        clean) would increase the metric.
    """
    n_layers = model.cfg.n_layers

    # Filter to only cache the residual stream activations we care about
    def names_filter(name: str) -> bool:
        return any(name == utils.get_act_name(position, layer=l) for l in range(n_layers))

    # Get clean activations (no grads needed)
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)

    # Get corrupted activations with gradients
    corrupted_cache, grad_dict = _get_cache_with_grads(
        model, corrupted_tokens, metric, names_filter=names_filter
    )

    # Compute attribution patching scores
    n_pos = corrupted_tokens.shape[-1]
    results = torch.zeros(n_layers, n_pos, device=model.cfg.device)

    for layer in range(n_layers):
        act_name = utils.get_act_name(position, layer=layer)
        # Difference: clean - corrupted
        diff = clean_cache[act_name] - corrupted_cache[act_name]
        # Gradient of metric w.r.t. this activation (evaluated at corrupted)
        grad = grad_dict[act_name]
        # Attribution: element-wise product summed over d_model, then mean over batch
        # Shape: [batch, pos, d_model] -> sum over d_model -> [batch, pos] -> mean over batch -> [pos]
        attr = (diff * grad).sum(dim=-1).mean(dim=0)
        results[layer] = attr

    return results


def attribution_patch_head_out(
    model: HookedTransformer,
    clean_tokens: Int[torch.Tensor, "batch pos"],
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
) -> Float[torch.Tensor, "n_layers n_heads"]:
    """Compute attribution patching scores for attention head outputs (across all positions).

    Approximates the effect of patching each attention head's output (the z vectors projected
    through W_O) from the corrupted run to the clean run.

    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokens for the clean run.
        corrupted_tokens: Tokens for the corrupted run.
        metric: A function from logits to a scalar metric.

    Returns:
        A tensor of shape [n_layers, n_heads] containing the approximate patching effect for
        each attention head (summed across all positions).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Cache z (pre-output) activations: shape [batch, pos, n_heads, d_head]
    def names_filter(name: str) -> bool:
        return any(name == utils.get_act_name("z", layer=l) for l in range(n_layers))

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)

    corrupted_cache, grad_dict = _get_cache_with_grads(
        model, corrupted_tokens, metric, names_filter=names_filter
    )

    results = torch.zeros(n_layers, n_heads, device=model.cfg.device)

    for layer in range(n_layers):
        act_name = utils.get_act_name("z", layer=layer)
        diff = clean_cache[act_name] - corrupted_cache[act_name]
        grad = grad_dict[act_name]
        # Shape: [batch, pos, n_heads, d_head]
        # Sum over d_head and pos, mean over batch -> [n_heads]
        attr = (diff * grad).sum(dim=-1).sum(dim=-2).mean(dim=0)
        results[layer] = attr

    return results


def attribution_patch_head_out_by_pos(
    model: HookedTransformer,
    clean_tokens: Int[torch.Tensor, "batch pos"],
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
) -> Float[torch.Tensor, "n_layers pos n_heads"]:
    """Compute attribution patching scores for attention head outputs at each position.

    Like :func:`attribution_patch_head_out` but preserves position information.

    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokens for the clean run.
        corrupted_tokens: Tokens for the corrupted run.
        metric: A function from logits to a scalar metric.

    Returns:
        A tensor of shape [n_layers, pos, n_heads] containing the approximate patching effect
        for each head at each position.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_pos = corrupted_tokens.shape[-1]

    def names_filter(name: str) -> bool:
        return any(name == utils.get_act_name("z", layer=l) for l in range(n_layers))

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)

    corrupted_cache, grad_dict = _get_cache_with_grads(
        model, corrupted_tokens, metric, names_filter=names_filter
    )

    results = torch.zeros(n_layers, n_pos, n_heads, device=model.cfg.device)

    for layer in range(n_layers):
        act_name = utils.get_act_name("z", layer=layer)
        diff = clean_cache[act_name] - corrupted_cache[act_name]
        grad = grad_dict[act_name]
        # Shape: [batch, pos, n_heads, d_head] -> sum d_head, mean batch -> [pos, n_heads]
        attr = (diff * grad).sum(dim=-1).mean(dim=0)
        results[layer] = attr

    return results


def attribution_patch_mlp_out(
    model: HookedTransformer,
    clean_tokens: Int[torch.Tensor, "batch pos"],
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
) -> Float[torch.Tensor, "n_layers pos"]:
    """Compute attribution patching scores for MLP layer outputs at each position.

    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokens for the clean run.
        corrupted_tokens: Tokens for the corrupted run.
        metric: A function from logits to a scalar metric.

    Returns:
        A tensor of shape [n_layers, pos] containing the approximate patching effect for each
        MLP layer at each position.
    """
    n_layers = model.cfg.n_layers

    def names_filter(name: str) -> bool:
        return any(name == utils.get_act_name("mlp_out", layer=l) for l in range(n_layers))

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)

    corrupted_cache, grad_dict = _get_cache_with_grads(
        model, corrupted_tokens, metric, names_filter=names_filter
    )

    n_pos = corrupted_tokens.shape[-1]
    results = torch.zeros(n_layers, n_pos, device=model.cfg.device)

    for layer in range(n_layers):
        act_name = utils.get_act_name("mlp_out", layer=layer)
        diff = clean_cache[act_name] - corrupted_cache[act_name]
        grad = grad_dict[act_name]
        # Shape: [batch, pos, d_model] -> sum d_model, mean batch -> [pos]
        attr = (diff * grad).sum(dim=-1).mean(dim=0)
        results[layer] = attr

    return results


def attribution_patch_block_every(
    model: HookedTransformer,
    clean_tokens: Int[torch.Tensor, "batch pos"],
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
) -> Float[torch.Tensor, "component_type n_layers pos"]:
    """Compute attribution patching for residual stream, attention output, and MLP output.

    This is the gradient-based analog of :func:`transformer_lens.patching.get_act_patch_block_every`.
    Returns a stacked tensor with attribution scores for resid_pre, attn_out, and mlp_out.

    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokens for the clean run.
        corrupted_tokens: Tokens for the corrupted run.
        metric: A function from logits to a scalar metric.

    Returns:
        A tensor of shape [3, n_layers, pos] where dim 0 indexes:
        [0] resid_pre, [1] attn_out, [2] mlp_out.
    """
    n_layers = model.cfg.n_layers

    # Cache all three activation types
    def names_filter(name: str) -> bool:
        for l in range(n_layers):
            if name in (
                utils.get_act_name("resid_pre", layer=l),
                utils.get_act_name("attn_out", layer=l),
                utils.get_act_name("mlp_out", layer=l),
            ):
                return True
        return False

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)

    corrupted_cache, grad_dict = _get_cache_with_grads(
        model, corrupted_tokens, metric, names_filter=names_filter
    )

    n_pos = corrupted_tokens.shape[-1]
    results = torch.zeros(3, n_layers, n_pos, device=model.cfg.device)

    for layer in range(n_layers):
        for i, act_type in enumerate(["resid_pre", "attn_out", "mlp_out"]):
            act_name = utils.get_act_name(act_type, layer=layer)
            diff = clean_cache[act_name] - corrupted_cache[act_name]
            grad = grad_dict[act_name]
            attr = (diff * grad).sum(dim=-1).mean(dim=0)
            results[i, layer] = attr

    return results


def attribution_patch_head_pattern(
    model: HookedTransformer,
    clean_tokens: Int[torch.Tensor, "batch pos"],
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
) -> Float[torch.Tensor, "n_layers n_heads"]:
    """Compute attribution patching scores for attention patterns (across all positions).

    Approximates the effect of patching each head's attention pattern from the corrupted run
    to the clean run.

    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokens for the clean run.
        corrupted_tokens: Tokens for the corrupted run.
        metric: A function from logits to a scalar metric.

    Returns:
        A tensor of shape [n_layers, n_heads] containing the approximate patching effect for
        each head's attention pattern (summed across all position pairs).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    def names_filter(name: str) -> bool:
        return any(name == utils.get_act_name("pattern", layer=l) for l in range(n_layers))

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)

    corrupted_cache, grad_dict = _get_cache_with_grads(
        model, corrupted_tokens, metric, names_filter=names_filter
    )

    results = torch.zeros(n_layers, n_heads, device=model.cfg.device)

    for layer in range(n_layers):
        act_name = utils.get_act_name("pattern", layer=layer)
        diff = clean_cache[act_name] - corrupted_cache[act_name]
        grad = grad_dict[act_name]
        # Shape: [batch, n_heads, dest_pos, src_pos]
        # Sum over both position dims, mean over batch -> [n_heads]
        attr = (diff * grad).sum(dim=-1).sum(dim=-1).mean(dim=0)
        results[layer] = attr

    return results


def attribution_patch_head_all_pos_every(
    model: HookedTransformer,
    clean_tokens: Int[torch.Tensor, "batch pos"],
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
) -> Float[torch.Tensor, "component_type n_layers n_heads"]:
    """Compute attribution patching for every head component type (output, Q, K, V, pattern).

    This is the gradient-based analog of
    :func:`transformer_lens.patching.get_act_patch_attn_head_all_pos_every`.

    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokens for the clean run.
        corrupted_tokens: Tokens for the corrupted run.
        metric: A function from logits to a scalar metric.

    Returns:
        A tensor of shape [5, n_layers, n_heads] where dim 0 indexes:
        [0] z (output), [1] q, [2] k, [3] v, [4] pattern.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    act_types = ["z", "q", "k", "v", "pattern"]

    def names_filter(name: str) -> bool:
        for l in range(n_layers):
            for act_type in act_types:
                if name == utils.get_act_name(act_type, layer=l):
                    return True
        return False

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)

    corrupted_cache, grad_dict = _get_cache_with_grads(
        model, corrupted_tokens, metric, names_filter=names_filter
    )

    results = torch.zeros(5, n_layers, n_heads, device=model.cfg.device)

    for layer in range(n_layers):
        for i, act_type in enumerate(act_types):
            act_name = utils.get_act_name(act_type, layer=layer)
            diff = clean_cache[act_name] - corrupted_cache[act_name]
            grad = grad_dict[act_name]

            if act_type == "pattern":
                # Shape: [batch, n_heads, dest_pos, src_pos]
                attr = (diff * grad).sum(dim=-1).sum(dim=-1).mean(dim=0)
            else:
                # Shape: [batch, pos, n_heads, d_head]
                attr = (diff * grad).sum(dim=-1).sum(dim=-2).mean(dim=0)

            # Handle models with different n_key_value_heads
            if attr.shape[0] < n_heads:
                padded = torch.zeros(n_heads, device=model.cfg.device)
                padded[: attr.shape[0]] = attr
                attr = padded

            results[i, layer] = attr

    return results
