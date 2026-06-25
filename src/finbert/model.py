"""
NanoFinBERT — A tiny transformer encoder trained from scratch on financial text.

Architecture overview:
  - 4 encoder layers
  - 128 hidden dimensions
  - 4 attention heads
  - 512 max sequence length
  - ~2M parameters (runs on CPU)

Inspired by Andrej Karpathy's nanoGPT: simple, readable, educational.
Each component is annotated to explain *why* it exists, not just *what* it does.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class FinancialEmbedding(nn.Module):
    """
    Combines learned token and positional embeddings (BERT-style).

    WHY positional information matters for financial text:
    In a sentence like "Tesla stock rose after earnings beat", the order of
    words carries meaning — "rose after" implies causality. Without positional
    information, the transformer sees a bag of tokens with no sense of order.

    WHY *learned* positions instead of fixed sinusoidal encodings:
    The token embeddings are initialised at a small scale (std=0.02 — the BERT
    scheme used throughout this model). Fixed sinusoidal encodings have a
    magnitude of ~1.0, roughly two orders of magnitude larger, so they swamp
    the token signal — the [CLS] position becomes identical for every input and
    the encoder goes effectively blind to which tokens are present (a
    representation collapse that pins accuracy at the majority-class baseline).
    Learned positional embeddings are initialised at the same 0.02 scale, so
    token and position contributions stay balanced and the model can actually
    read the text. This matches real BERT (Devlin et al., 2019), which also
    uses learned position embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Learned embedding matrix: maps each token ID to a dense vector.
        # Shape: (vocab_size, hidden_dim)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        # Learned positional embedding — one vector per absolute position.
        # Initialised at the same 0.02 scale as the token embedding (see
        # NanoFinBERT._init_weights), keeping the two signals balanced.
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Layer normalisation applied after combining embeddings and positions.
        # Stabilises training by normalising activations to zero-mean, unit-variance.
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Constant position ids [0, 1, ..., max_seq_len-1], sliced per forward.
        # Registered as a buffer so it moves with the model's device but is not trained.
        self.register_buffer(
            "position_ids", torch.arange(max_seq_len).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) integer token IDs

        Returns:
            embeddings: (batch, seq_len, hidden_dim)
        """
        seq_len = input_ids.size(1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(self.position_ids[:, :seq_len])

        # Add positional embedding (broadcasts over the batch dimension).
        x = token_emb + pos_emb
        return self.dropout(self.layer_norm(x))


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention.

    HOW attention captures entity relationships in financial text:
    When processing "Tesla reported a 30% earnings beat", the attention
    mechanism lets the model learn that "reported" and "beat" should
    attend strongly to "Tesla" — linking the entity to its event. Multiple
    heads allow the model to capture different relationship types
    simultaneously (e.g., one head for subject-verb, another for
    company-metric relationships).

    Scaled dot-product attention formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    The sqrt(d_k) scaling prevents the dot products from growing too large,
    which would push softmax into regions with vanishing gradients.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017), Section 3.2.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # dimension per attention head

        # Projections for queries, keys, values, and output.
        # We use a single matrix for efficiency, then split by head.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape

        # Project and reshape into (batch, num_heads, seq_len, head_dim)
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(self.q_proj(x))
        k = reshape(self.k_proj(x))
        v = reshape(self.v_proj(x))

        # Scaled dot-product attention scores: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Mask padding positions so they don't contribute to attention.
        if attention_mask is not None:
            # Expand mask to (batch, 1, 1, seq_len) for broadcasting.
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: (batch, heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v)

        # Merge heads back: (batch, seq_len, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network with GELU activation.

    ROLE of the FFN after attention:
    Attention is inherently a linear operation (weighted average of values).
    The FFN adds non-linearity, allowing the model to learn complex
    transformations of the attended representations — for example, mapping
    "positive sentiment about earnings" into a feature that robustly
    activates the SentimentHead's "positive" class.

    We expand to 4x the hidden dimension (standard transformer convention)
    to give the model a larger representational capacity per layer before
    projecting back.

    GELU (Gaussian Error Linear Unit) is preferred over ReLU in modern
    transformers because it has a smooth gradient near zero, which aids
    training stability and produces slightly better downstream performance.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        ffn_dim = hidden_dim * 4  # 4× expansion (standard transformer ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder Layer (Attention + FFN + Residuals + LayerNorm)
# ---------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    """
    A single transformer encoder block.

    Each block applies:
      1. Multi-head self-attention (with residual connection + LayerNorm)
      2. Feed-forward network      (with residual connection + LayerNorm)

    Residual connections (x = x + sublayer(x)) allow gradients to flow
    directly from the output back to earlier layers — solving the vanishing
    gradient problem that plagued deep RNNs before transformers.

    Pre-layer normalisation (applied before each sublayer) is used here
    as it leads to more stable training than post-LN for small models.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention with residual
        x = x + self.attention(self.norm1(x), attention_mask)
        # Pre-LN FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Classification Heads
# ---------------------------------------------------------------------------


class SentimentHead(nn.Module):
    """
    Classifies the [CLS] token representation into 3 sentiment classes:
    0 = negative, 1 = neutral, 2 = positive.

    The [CLS] token (always at position 0) accumulates global sentence
    information through self-attention across the entire sequence, making
    it a natural anchor for sentence-level classification tasks.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_representation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_representation: (batch, hidden_dim) — the [CLS] token output

        Returns:
            logits: (batch, num_classes)
        """
        return self.classifier(cls_representation)


# ---------------------------------------------------------------------------
# Full NanoFinBERT Model
# ---------------------------------------------------------------------------


class NanoFinBERT(nn.Module):
    """
    NanoFinBERT: A tiny BERT-style transformer encoder for financial NLP.

    Design choices explained:
    - vocab_size=8000:  Small vocabulary trades coverage for speed.
                        Financial text has a predictable lexicon (tickers,
                        financial terms, company names).
    - hidden_dim=128:   Small enough to train on CPU, large enough to
                        capture meaningful representations.
    - num_layers=4:     BERT-base uses 12; we use 4 for educational clarity
                        and CPU-friendliness.
    - num_heads=4:      head_dim = 32, giving fine-grained multi-head signals
                        without excessive compute.
    - max_seq_len=256:  Financial headlines and short reports are < 256 tokens.
    - Total params:     ~2M — comparable to reading a few pages of documentation.

    The model outputs:
      - sentiment_logits: (batch, 3) for positive/negative/neutral classification
      - last_hidden_state: (batch, seq_len, hidden_dim) full encoder output
      - pooled_output: (batch, hidden_dim) [CLS] representation
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.config = {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }

        # Embedding layer (token + positional)
        self.embedding = FinancialEmbedding(vocab_size, hidden_dim, max_seq_len, dropout)

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Final layer normalisation (applied to encoder output)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Pooler: transforms [CLS] output for classification tasks.
        # BERT uses a tanh-activated linear layer here to project into
        # a "pooled" space better suited for sentence-level tasks.
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Task-specific head
        self.sentiment_head = SentimentHead(hidden_dim, num_classes=3)

        # Initialise weights following BERT's initialisation scheme.
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise parameters using truncated normal distribution (std=0.02).
        This matches the original BERT implementation and helps with stable
        training from scratch.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the full model.

        Args:
            input_ids:      (batch, seq_len) integer token IDs
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

        Returns:
            dict with keys:
                - "sentiment_logits":  (batch, 3)
                - "last_hidden_state": (batch, seq_len, hidden_dim)
                - "pooled_output":     (batch, hidden_dim)
        """
        # Step 1: Embed tokens + add positional information
        x = self.embedding(input_ids)

        # Step 2: Pass through each encoder layer
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # Step 3: Final normalisation
        x = self.final_norm(x)

        # Step 4: Extract [CLS] token (position 0) for sentence-level tasks
        cls_output = x[:, 0, :]  # (batch, hidden_dim)
        pooled = self.pooler(cls_output)

        # Step 5: Classify sentiment
        sentiment_logits = self.sentiment_head(pooled)

        return {
            "sentiment_logits": sentiment_logits,
            "last_hidden_state": x,
            "pooled_output": pooled,
        }

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Return a breakdown of parameters by component."""
        breakdown = {}
        for name, module in self.named_children():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            breakdown[name] = count
        return breakdown

    @classmethod
    def from_checkpoint(cls, path: str) -> "NanoFinBERT":
        """Load model from a saved checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
