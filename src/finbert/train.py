"""
Training loop for NanoFinBERT — educational, heavily commented.

Inspired by nanoGPT's training philosophy: readable > clever.

Training recipe:
  - AdamW optimiser (decoupled weight decay, standard for transformers)
  - Cosine learning rate schedule with linear warmup
  - Gradient clipping to prevent exploding gradients
  - Checkpoint saving every N steps
  - Validation evaluation every epoch

Usage:
    from finbert.model import NanoFinBERT
    from finbert.tokenizer import FinancialTokenizer, build_default_tokenizer
    from finbert.dataset import load_sample_dataset, make_dataloaders
    from finbert.train import TrainConfig, Trainer

    tokenizer = build_default_tokenizer(texts)
    dataset = load_sample_dataset(tokenizer)
    train_loader, val_loader = make_dataloaders(dataset)

    model = NanoFinBERT()
    config = TrainConfig(num_epochs=10, lr=3e-4)
    trainer = Trainer(model, tokenizer, train_loader, val_loader, config)
    trainer.train()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from finbert.model import NanoFinBERT

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    All hyperparameters for a training run, kept in one place.

    WHY dataclass: makes it trivial to serialise, log, and reproduce runs.
    Every hyperparameter should be explicit — no magic constants buried in code.
    """

    # Optimiser
    lr: float = 3e-4  # Peak learning rate (AdamW, transformer sweet-spot)
    weight_decay: float = 0.01  # L2 regularisation on non-bias parameters
    beta1: float = 0.9  # AdamW momentum for first moment
    beta2: float = 0.999  # AdamW momentum for second moment
    eps: float = 1e-8  # Numerical stability for AdamW

    # Schedule
    warmup_steps: int = 100  # Linear warmup before cosine decay begins
    num_epochs: int = 10  # Total training epochs

    # Regularisation
    grad_clip: float = 1.0  # Max gradient norm (clips exploding gradients)

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 200  # Save a checkpoint every N training steps
    eval_every_n_epochs: int = 1  # Evaluate on val set every N epochs

    # Logging
    log_every_n_steps: int = 10  # Print training stats every N steps

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def cosine_schedule_with_warmup(
    optimiser: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay.

    WHY warmup: At initialisation, the model's gradients are large and noisy.
    Warming up the LR from 0 prevents destructive early updates.

    WHY cosine decay: Smooth annealing lets the optimiser settle into a good
    minimum near the end of training, unlike step decay which causes sudden drops.

    LR schedule shape:
        step < warmup_steps  →  lr * (step / warmup_steps)
        step >= warmup_steps →  lr * 0.5 * (1 + cos(π * progress))
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay from warmup_steps to total_steps
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimiser, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Educational training loop for NanoFinBERT.

    Handles:
      - Training step (forward + backward + optimiser step)
      - Validation evaluation
      - Gradient norm tracking
      - Checkpoint saving / loading
      - Basic progress logging

    Args:
        model:        NanoFinBERT instance (untrained weights are fine to start).
        tokenizer:    FinancialTokenizer (used for saving alongside checkpoints).
        train_loader: DataLoader for training set.
        val_loader:   DataLoader for validation set.
        config:       TrainConfig with all hyperparameters.
    """

    def __init__(
        self,
        model: NanoFinBERT,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
    ) -> None:
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # AdamW: Adam with decoupled weight decay.
        # We do NOT apply weight decay to biases or LayerNorm parameters
        # (common best practice — they are scale parameters, not weights).
        decay_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in ["bias", "norm"])
        ]
        no_decay_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and any(nd in n for nd in ["bias", "norm"])
        ]
        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimiser = AdamW(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        # Compute total training steps for the LR schedule
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.num_epochs

        self.scheduler = cosine_schedule_with_warmup(
            self.optimiser, config.warmup_steps, total_steps
        )

        # Cross-entropy loss for 3-class sentiment classification
        self.criterion = nn.CrossEntropyLoss()

        # Tracking. Start best accuracy below zero so the first evaluation always
        # writes a best_model checkpoint — otherwise a model that never beats a
        # 0.0 validation accuracy (possible on tiny/degenerate runs) would finish
        # training with no saved checkpoint at all.
        self.global_step = 0
        self.best_val_accuracy = -1.0
        self.history: list[dict] = []  # [{epoch, train_loss, val_loss, val_acc}, ...]

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, num_epochs: int | None = None) -> list[dict]:
        """
        Run the full training loop.

        Args:
            num_epochs: Override config.num_epochs if provided.

        Returns:
            Training history as a list of per-epoch dicts.
        """
        epochs = num_epochs or self.config.num_epochs
        total_params = self.model.count_parameters()

        print("=" * 60)
        print("NanoFinBERT Training")
        print("=" * 60)
        print(f"  Parameters:   {total_params:,}")
        print(f"  Device:       {self.config.device}")
        print(f"  Epochs:       {epochs}")
        print(f"  Batch size:   {self.train_loader.batch_size}")
        print(f"  Train steps:  {len(self.train_loader) * epochs:,}")
        print(f"  Peak LR:      {self.config.lr}")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss = self._train_epoch(epoch)

            val_metrics = {}
            if epoch % self.config.eval_every_n_epochs == 0:
                val_metrics = self.evaluate()
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]

                # Save best model based on validation accuracy
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.save_checkpoint(
                        str(Path(self.config.checkpoint_dir) / "best_model.pt"),
                        epoch=epoch,
                    )

                epoch_time = time.time() - epoch_start
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.3f} | "
                    f"time={epoch_time:.1f}s"
                )
            else:
                epoch_time = time.time() - epoch_start
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"time={epoch_time:.1f}s"
                )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
            }
            self.history.append(record)

        print("=" * 60)
        # Clamp the -1.0 sentinel (no evaluation ran) to 0.0 for display.
        print(f"Training complete. Best val accuracy: {max(0.0, self.best_val_accuracy):.3f}")
        return self.history

    def _train_epoch(self, epoch: int) -> float:
        """Run one epoch of training. Returns mean training loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            loss, grad_norm = self._train_step(batch)
            total_loss += loss
            n_batches += 1

            # Log progress
            if self.global_step % self.config.log_every_n_steps == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  step={self.global_step:5d} | "
                    f"loss={loss:.4f} | "
                    f"grad_norm={grad_norm:.3f} | "
                    f"lr={current_lr:.2e}"
                )

            # Save periodic checkpoint
            if (
                self.config.save_every_n_steps > 0
                and self.global_step % self.config.save_every_n_steps == 0
                and self.global_step > 0
            ):
                ckpt_path = Path(self.config.checkpoint_dir) / f"step_{self.global_step:06d}.pt"
                self.save_checkpoint(str(ckpt_path), epoch=epoch)

        return total_loss / max(n_batches, 1)

    def _train_step(self, batch: dict) -> tuple[float, float]:
        """
        Single training step: forward → loss → backward → clip → step.

        Returns:
            (loss_value, gradient_norm)
        """
        device = self.config.device

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        logits = outputs["sentiment_logits"]  # (batch, 3)

        # Cross-entropy loss: measures how far predictions are from true labels.
        # It averages over the batch, so loss is batch-size independent.
        loss = self.criterion(logits, labels)

        # Backward pass: compute gradients for all parameters via autograd.
        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping: prevents gradient explosion in early training.
        # If the global L2 norm exceeds grad_clip, scale all gradients down.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        ).item()

        # Optimiser step: update parameters using computed gradients.
        self.optimiser.step()

        # LR scheduler step: advance the cosine schedule by one step.
        self.scheduler.step()

        self.global_step += 1
        return loss.item(), grad_norm

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the validation set.

        Returns:
            dict with "loss" and "accuracy".
        """
        self.model.eval()
        device = self.config.device

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = self.model(input_ids, attention_mask)
                logits = outputs["sentiment_logits"]

                loss = self.criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)

                # Accuracy: fraction of correct predictions
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        self.model.train()  # Return to training mode
        return {"loss": avg_loss, "accuracy": accuracy}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """
        Save model weights, optimiser state, and config.

        Saving the optimiser state lets you resume training exactly where
        you left off — the momentum buffers are preserved.
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.model.config,
            "best_val_accuracy": self.best_val_accuracy,
            "history": self.history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Resume training from a checkpoint. Returns the epoch to resume from.
        """
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        self.history = checkpoint.get("history", [])
        return checkpoint.get("epoch", 0)
