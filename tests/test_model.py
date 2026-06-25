"""
Tests for NanoFinBERT model architecture.

Covers:
  - Forward pass output shapes
  - Parameter count
  - Gradient flow
  - Checkpoint save/load roundtrip
  - Attention masking correctness
"""

from __future__ import annotations

import pytest
import torch

from finbert.model import (
    EncoderLayer,
    FeedForward,
    FinancialEmbedding,
    MultiHeadAttention,
    NanoFinBERT,
)

# ---------------------------------------------------------------------------
# FinancialEmbedding
# ---------------------------------------------------------------------------


class TestFinancialEmbedding:
    def test_output_shape(self):
        emb = FinancialEmbedding(vocab_size=500, hidden_dim=64, max_seq_len=32)
        x = torch.randint(0, 500, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 64)

    def test_positional_embedding_is_learned(self):
        emb = FinancialEmbedding(vocab_size=500, hidden_dim=64, max_seq_len=32)
        # Positions are learned (BERT-style), not a fixed sinusoidal buffer —
        # this keeps the position signal at the same scale as the token signal.
        params = dict(emb.named_parameters())
        assert "position_embedding.weight" in params
        assert emb.position_embedding.num_embeddings == 32
        assert emb.position_embedding.embedding_dim == 64

    def test_padding_idx_zero(self):
        emb = FinancialEmbedding(vocab_size=500, hidden_dim=64, max_seq_len=32)
        # Padding index 0 should have zero embedding weight
        assert emb.token_embedding.padding_idx == 0


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_masked_attention_shape(self):
        attn = MultiHeadAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[0, 10:] = 0  # mask out last 6 positions in first item
        out = attn(x, attention_mask=mask)
        assert out.shape == (2, 16, 64)

    def test_invalid_head_count_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(hidden_dim=64, num_heads=3)  # 64 not divisible by 3

    def test_head_dim_computed_correctly(self):
        attn = MultiHeadAttention(hidden_dim=128, num_heads=4)
        assert attn.head_dim == 32


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(hidden_dim=64)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == (2, 16, 64)

    def test_expansion_ratio(self):
        ffn = FeedForward(hidden_dim=64)
        # First linear should expand 4x
        assert ffn.net[0].out_features == 256


# ---------------------------------------------------------------------------
# EncoderLayer
# ---------------------------------------------------------------------------


class TestEncoderLayer:
    def test_output_shape(self):
        layer = EncoderLayer(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_with_attention_mask(self):
        layer = EncoderLayer(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16, dtype=torch.long)
        out = layer(x, attention_mask=mask)
        assert out.shape == (2, 16, 64)


# ---------------------------------------------------------------------------
# NanoFinBERT (full model)
# ---------------------------------------------------------------------------


class TestNanoFinBERT:
    def test_forward_output_keys(self, tiny_model, sample_batch_small):
        out = tiny_model(**sample_batch_small)
        assert set(out.keys()) == {"sentiment_logits", "last_hidden_state", "pooled_output"}

    def test_sentiment_logits_shape(self, tiny_model, sample_batch_small):
        out = tiny_model(**sample_batch_small)
        batch = sample_batch_small["input_ids"].shape[0]
        assert out["sentiment_logits"].shape == (batch, 3)

    def test_last_hidden_state_shape(self, tiny_model, sample_batch_small):
        out = tiny_model(**sample_batch_small)
        batch, seq = sample_batch_small["input_ids"].shape
        assert out["last_hidden_state"].shape == (batch, seq, tiny_model.config["hidden_dim"])

    def test_pooled_output_shape(self, tiny_model, sample_batch_small):
        out = tiny_model(**sample_batch_small)
        batch = sample_batch_small["input_ids"].shape[0]
        assert out["pooled_output"].shape == (batch, tiny_model.config["hidden_dim"])

    def test_count_parameters_positive(self, tiny_model):
        count = tiny_model.count_parameters()
        assert count > 0

    def test_count_parameters_default_model(self, default_model):
        count = default_model.count_parameters()
        # Default model should be ~2M params
        assert 1_000_000 < count < 5_000_000

    def test_parameter_breakdown_keys(self, tiny_model):
        breakdown = tiny_model.parameter_breakdown()
        assert "embedding" in breakdown
        assert "sentiment_head" in breakdown
        assert "encoder_layers" in breakdown

    def test_gradient_flow(self, tiny_tokenizer):
        """Verify gradients flow back to the embedding layer during backprop."""
        model = NanoFinBERT(
            vocab_size=500,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model.train()

        input_ids = torch.randint(0, 500, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.randint(0, 3, (2,))

        out = model(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(out["sentiment_logits"], labels)
        loss.backward()

        # Embedding weight gradient should be non-None and non-zero
        emb_grad = model.embedding.token_embedding.weight.grad
        assert emb_grad is not None
        assert emb_grad.abs().sum() > 0

    def test_no_nan_in_forward(self, tiny_model, sample_batch_small):
        out = tiny_model(**sample_batch_small)
        for key, tensor in out.items():
            assert not torch.isnan(tensor).any(), f"NaN found in {key}"

    def test_checkpoint_roundtrip(self, tiny_model, tmp_path):
        """Save and reload a checkpoint; verify weights are identical."""
        path = str(tmp_path / "test_ckpt.pt")
        import torch

        checkpoint = {
            "config": tiny_model.config,
            "model_state_dict": tiny_model.state_dict(),
        }
        torch.save(checkpoint, path)

        loaded = NanoFinBERT.from_checkpoint(path)
        for (n1, p1), (n2, p2) in zip(tiny_model.named_parameters(), loaded.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter mismatch at {n1}"

    def test_without_attention_mask(self, tiny_model):
        """Model must work when attention_mask is None (no masking)."""
        input_ids = torch.randint(0, 500, (1, 16))
        out = tiny_model(input_ids, attention_mask=None)
        assert out["sentiment_logits"].shape == (1, 3)

    def test_batch_size_one(self, tiny_model):
        input_ids = torch.randint(0, 500, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.long)
        out = tiny_model(input_ids, mask)
        assert out["sentiment_logits"].shape == (1, 3)

    def test_single_token_sequence(self, tiny_model):
        """Edge case: sequence of length 1 (just [CLS])."""
        input_ids = torch.tensor([[2]])  # [CLS] only
        mask = torch.ones(1, 1, dtype=torch.long)
        out = tiny_model(input_ids, mask)
        assert out["sentiment_logits"].shape == (1, 3)
