"""Tests for attribution patching (gradient-based activation patching approximation)."""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.attribution_patching import (
    attribution_patch_block_every,
    attribution_patch_head_out,
    attribution_patch_head_out_by_pos,
    attribution_patch_head_pattern,
    attribution_patch_mlp_out,
    attribution_patch_residual_stream,
)


@pytest.fixture(scope="module")
def model():
    return HookedTransformer.from_pretrained("solu-1l", device="cpu")


@pytest.fixture(scope="module")
def clean_tokens(model):
    return model.to_tokens("The cat sat on")


@pytest.fixture(scope="module")
def corrupted_tokens(model):
    return model.to_tokens("The dog ran on")


def metric_fn(logits):
    """Simple metric: sum of logits at last position."""
    return logits[:, -1, :].sum()


class TestAttributionPatchResidualStream:
    def test_shape(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_residual_stream(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        n_pos = clean_tokens.shape[-1]
        assert result.shape == (model.cfg.n_layers, n_pos)

    def test_values_finite(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_residual_stream(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        assert torch.isfinite(result).all()

    def test_identical_inputs_give_zero(self, model, clean_tokens):
        """When clean == corrupted, attribution scores should be zero."""
        result = attribution_patch_residual_stream(
            model, clean_tokens, clean_tokens, metric_fn
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)


class TestAttributionPatchHeadOut:
    def test_shape(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_head_out(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        assert result.shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_values_finite(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_head_out(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        assert torch.isfinite(result).all()

    def test_identical_inputs_give_zero(self, model, clean_tokens):
        result = attribution_patch_head_out(model, clean_tokens, clean_tokens, metric_fn)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)


class TestAttributionPatchHeadOutByPos:
    def test_shape(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_head_out_by_pos(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        n_pos = clean_tokens.shape[-1]
        assert result.shape == (model.cfg.n_layers, n_pos, model.cfg.n_heads)

    def test_sum_matches_head_out(self, model, clean_tokens, corrupted_tokens):
        """Summing over positions should approximately match attribution_patch_head_out."""
        by_pos = attribution_patch_head_out_by_pos(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        all_pos = attribution_patch_head_out(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        # Sum over position dim
        summed = by_pos.sum(dim=1)
        assert torch.allclose(summed, all_pos, atol=1e-4)


class TestAttributionPatchMlpOut:
    def test_shape(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_mlp_out(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        n_pos = clean_tokens.shape[-1]
        assert result.shape == (model.cfg.n_layers, n_pos)

    def test_values_finite(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_mlp_out(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        assert torch.isfinite(result).all()


class TestAttributionPatchBlockEvery:
    def test_shape(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_block_every(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        n_pos = clean_tokens.shape[-1]
        assert result.shape == (3, model.cfg.n_layers, n_pos)

    def test_components_match_individual(self, model, clean_tokens, corrupted_tokens):
        """The block_every results should match individual function results."""
        block_results = attribution_patch_block_every(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        resid_results = attribution_patch_residual_stream(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        mlp_results = attribution_patch_mlp_out(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        assert torch.allclose(block_results[0], resid_results, atol=1e-4)
        assert torch.allclose(block_results[2], mlp_results, atol=1e-4)


class TestAttributionPatchHeadPattern:
    def test_shape(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_head_pattern(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        assert result.shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_values_finite(self, model, clean_tokens, corrupted_tokens):
        result = attribution_patch_head_pattern(
            model, clean_tokens, corrupted_tokens, metric_fn
        )
        assert torch.isfinite(result).all()
