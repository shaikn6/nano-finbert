"""
Extended model tests to cover branches missed by the existing test_model.py.
Targets: _init_weights edge cases, from_checkpoint error path, model repr/config,
full-size batch, padding_idx zeroing, attention scale, multiple-batch sizes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from finbert.model import (
    EncoderLayer,
    FeedForward,
    FinancialEmbedding,
    MultiHeadAttention,
    NanoFinBERT,
    SentimentHead,
)

# ---------------------------------------------------------------------------
# SentimentHead
# ---------------------------------------------------------------------------


class TestSentimentHead:
    def test_output_shape_default_classes(self):
        head = SentimentHead(hidden_dim=64)
        x = torch.randn(3, 64)
        out = head(x)
        assert out.shape == (3, 3)

    def test_output_shape_custom_classes(self):
        head = SentimentHead(hidden_dim=64, num_classes=5)
        x = torch.randn(2, 64)
        out = head(x)
        assert out.shape == (2, 5)

    def test_no_nan_in_output(self):
        head = SentimentHead(hidden_dim=64)
        x = torch.randn(4, 64)
        out = head(x)
        assert not torch.isnan(out).any()

    def test_forward_is_differentiable(self):
        head = SentimentHead(hidden_dim=32)
        x = torch.randn(2, 32, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# FinancialEmbedding — additional branches
# ---------------------------------------------------------------------------


class TestFinancialEmbeddingExtra:
    def test_padding_idx_row_zeroed_after_init(self):
        """After NanoFinBERT._init_weights, padding row should be zero."""
        model = NanoFinBERT(
            vocab_size=100, hidden_dim=32, num_layers=1, num_heads=2, max_seq_len=16
        )
        pad_row = model.embedding.token_embedding.weight.data[0]
        assert torch.all(pad_row == 0.0)

    def test_position_embedding_shape(self):
        emb = FinancialEmbedding(vocab_size=200, hidden_dim=64, max_seq_len=128)
        # Learned positional embedding table: (max_seq_len, hidden_dim).
        assert emb.position_embedding.weight.shape == (128, 64)
        # position_ids buffer covers every position.
        assert dict(emb.named_buffers())["position_ids"].shape == (1, 128)

    def test_different_seq_lengths_work(self):
        emb = FinancialEmbedding(vocab_size=200, hidden_dim=64, max_seq_len=128)
        for seq in (1, 8, 64, 128):
            x = torch.randint(0, 200, (1, seq))
            out = emb(x)
            assert out.shape == (1, seq, 64)

    def test_dropout_zero_preserves_values_in_eval(self):
        """With dropout=0, eval mode should not change output."""
        emb = FinancialEmbedding(vocab_size=100, hidden_dim=32, max_seq_len=16, dropout=0.0)
        emb.eval()
        x = torch.randint(0, 100, (1, 8))
        out1 = emb(x)
        out2 = emb(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# MultiHeadAttention — additional branches
# ---------------------------------------------------------------------------


class TestMultiHeadAttentionExtra:
    def test_scale_is_sqrt_head_dim(self):
        import math

        attn = MultiHeadAttention(hidden_dim=128, num_heads=4)
        assert attn.scale == pytest.approx(math.sqrt(32))

    def test_all_zeros_attention_mask_produces_uniform_attention(self):
        """All-zero mask (all padding) leads to uniform attention via -inf → softmax."""
        attn = MultiHeadAttention(hidden_dim=64, num_heads=4, dropout=0.0)
        attn.eval()
        x = torch.randn(1, 4, 64)
        mask = torch.zeros(1, 4, dtype=torch.long)
        # Should not crash (softmax over -inf in each column produces nan/uniform;
        # we just verify shape and that no Python exception is raised)
        out = attn(x, attention_mask=mask)
        assert out.shape == (1, 4, 64)

    def test_batch_size_eight(self):
        attn = MultiHeadAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(8, 32, 64)
        out = attn(x)
        assert out.shape == (8, 32, 64)

    def test_projections_have_no_bias(self):
        attn = MultiHeadAttention(hidden_dim=64, num_heads=4)
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None

    def test_out_proj_has_bias(self):
        attn = MultiHeadAttention(hidden_dim=64, num_heads=4)
        assert attn.out_proj.bias is not None


# ---------------------------------------------------------------------------
# FeedForward — extra
# ---------------------------------------------------------------------------


class TestFeedForwardExtra:
    def test_gelu_activation_present(self):
        ffn = FeedForward(hidden_dim=64)
        # net[1] should be GELU
        assert isinstance(ffn.net[1], nn.GELU)

    def test_two_dropout_layers(self):
        ffn = FeedForward(hidden_dim=64)
        dropout_layers = [m for m in ffn.net if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 2

    def test_second_linear_projects_back_to_hidden(self):
        ffn = FeedForward(hidden_dim=64)
        # net[-1] is Dropout; the projection back to hidden is the last Linear (net[-2]).
        assert ffn.net[-2].out_features == 64

    def test_large_hidden_dim(self):
        ffn = FeedForward(hidden_dim=512)
        x = torch.randn(2, 10, 512)
        out = ffn(x)
        assert out.shape == (2, 10, 512)


# ---------------------------------------------------------------------------
# EncoderLayer — extra
# ---------------------------------------------------------------------------


class TestEncoderLayerExtra:
    def test_residual_connection_changes_input(self):
        """Output should differ from input (residuals add on top)."""
        torch.manual_seed(0)
        layer = EncoderLayer(hidden_dim=64, num_heads=4, dropout=0.0)
        layer.eval()
        x = torch.randn(1, 8, 64)
        out = layer(x)
        assert not torch.allclose(x, out)

    def test_has_two_layer_norms(self):
        layer = EncoderLayer(hidden_dim=64, num_heads=4)
        norms = [m for m in layer.modules() if isinstance(m, nn.LayerNorm)]
        assert len(norms) >= 2


# ---------------------------------------------------------------------------
# NanoFinBERT — additional coverage
# ---------------------------------------------------------------------------


class TestNanoFinBERTExtra:
    def test_config_dict_has_all_keys(self):
        model = NanoFinBERT()
        for key in (
            "vocab_size",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "max_seq_len",
            "dropout",
        ):
            assert key in model.config

    def test_default_config_values(self):
        model = NanoFinBERT()
        assert model.config["vocab_size"] == 8000
        assert model.config["hidden_dim"] == 128
        assert model.config["num_layers"] == 4
        assert model.config["num_heads"] == 4
        assert model.config["max_seq_len"] == 256

    def test_pooler_uses_tanh(self):
        model = NanoFinBERT(
            vocab_size=100, hidden_dim=32, num_layers=1, num_heads=2, max_seq_len=16
        )
        has_tanh = any(isinstance(m, nn.Tanh) for m in model.pooler.modules())
        assert has_tanh

    def test_encoder_layers_count(self):
        model = NanoFinBERT(
            vocab_size=100, hidden_dim=32, num_layers=3, num_heads=2, max_seq_len=16
        )
        assert len(model.encoder_layers) == 3

    def test_final_norm_is_layer_norm(self):
        model = NanoFinBERT(
            vocab_size=100, hidden_dim=32, num_layers=1, num_heads=2, max_seq_len=16
        )
        assert isinstance(model.final_norm, nn.LayerNorm)

    def test_parameter_breakdown_sums_to_total(self):
        model = NanoFinBERT(
            vocab_size=500, hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=32
        )
        breakdown = model.parameter_breakdown()
        total = sum(breakdown.values())
        assert total == model.count_parameters()

    def test_forward_sentiment_logits_sum_not_zero(self):
        model = NanoFinBERT(
            vocab_size=500, hidden_dim=64, num_layers=1, num_heads=2, max_seq_len=32
        )
        model.eval()
        ids = torch.randint(0, 500, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        out = model(ids, mask)
        assert out["sentiment_logits"].abs().sum() > 0

    def test_pooled_output_bounded_by_tanh(self):
        """Pooler uses Tanh, so pooled_output values should be in (-1, 1)."""
        model = NanoFinBERT(
            vocab_size=500, hidden_dim=64, num_layers=1, num_heads=2, max_seq_len=32
        )
        model.eval()
        ids = torch.randint(0, 500, (1, 8))
        out = model(ids)
        # tanh output in (-1, 1) — check no value is outside
        assert out["pooled_output"].abs().max().item() <= 1.0 + 1e-5

    def test_from_checkpoint_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            NanoFinBERT.from_checkpoint(str(tmp_path / "nonexistent.pt"))

    def test_from_checkpoint_with_full_config(self, tmp_path):
        model = NanoFinBERT(
            vocab_size=200, hidden_dim=32, num_layers=1, num_heads=2, max_seq_len=16
        )
        path = str(tmp_path / "full.pt")
        import torch as _torch

        _torch.save({"config": model.config, "model_state_dict": model.state_dict()}, path)
        loaded = NanoFinBERT.from_checkpoint(path)
        assert loaded.config == model.config

    def test_large_batch_forward(self):
        model = NanoFinBERT(
            vocab_size=500, hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=64
        )
        model.eval()
        ids = torch.randint(0, 500, (32, 64))
        mask = torch.ones(32, 64, dtype=torch.long)
        out = model(ids, mask)
        assert out["sentiment_logits"].shape == (32, 3)

    def test_no_inf_in_forward_with_partial_mask(self):
        """Partial masking should not produce inf in output."""
        model = NanoFinBERT(
            vocab_size=500, hidden_dim=64, num_layers=1, num_heads=2, max_seq_len=32
        )
        model.eval()
        ids = torch.randint(0, 500, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[0, 10:] = 0  # partially mask first item
        out = model(ids, mask)
        for key, val in out.items():
            assert not torch.isinf(val).any(), f"Inf in {key}"

    def test_init_weights_linear_weight_std(self):
        """Linear weights should be initialised with std ~0.02."""
        torch.manual_seed(42)
        model = NanoFinBERT(
            vocab_size=500, hidden_dim=128, num_layers=2, num_heads=4, max_seq_len=64
        )
        # Collect std of linear weights
        stds = [m.weight.data.std().item() for m in model.modules() if isinstance(m, nn.Linear)]
        # All stds should be roughly in [0.005, 0.1] (trunc_normal with std=0.02)
        for std in stds:
            assert 0.001 < std < 0.15, f"Unexpected weight std: {std}"

    def test_layer_norm_init_weight_one_bias_zero(self):
        model = NanoFinBERT(
            vocab_size=200, hidden_dim=32, num_layers=1, num_heads=2, max_seq_len=16
        )
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                assert torch.all(m.weight.data == 1.0)
                assert torch.all(m.bias.data == 0.0)
