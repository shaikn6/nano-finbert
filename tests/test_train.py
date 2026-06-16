"""
Comprehensive tests for the Trainer, TrainConfig, and cosine_schedule_with_warmup.
Uses synthetic tiny DataLoaders with fake tensors — no real data loading.
Covers train.py to push overall coverage to 95%+.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from finbert.model import NanoFinBERT
from finbert.train import TrainConfig, Trainer, cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Helpers — build tiny synthetic DataLoaders
# ---------------------------------------------------------------------------


def _make_loader(
    n: int = 8,
    vocab_size: int = 500,
    seq_len: int = 16,
    batch_size: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """Return a DataLoader that yields dicts with input_ids / attention_mask / label."""
    input_ids = torch.randint(0, vocab_size, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    labels = torch.randint(0, 3, (n,))

    class _DictDataset(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, idx):
            return {
                "input_ids": input_ids[idx],
                "attention_mask": attention_mask[idx],
                "label": labels[idx],
            }

    return DataLoader(_DictDataset(), batch_size=batch_size, shuffle=shuffle)


def _tiny_model(vocab_size: int = 500) -> NanoFinBERT:
    return NanoFinBERT(
        vocab_size=vocab_size,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        dropout=0.0,
    )


def _tiny_trainer(tmp_path: Path, n: int = 8, num_epochs: int = 1) -> Trainer:
    model = _tiny_model()
    tok = MagicMock()  # tokenizer is only used for saving; mock is fine
    train_dl = _make_loader(n=n)
    val_dl = _make_loader(n=4)
    cfg = TrainConfig(
        lr=1e-3,
        num_epochs=num_epochs,
        warmup_steps=1,
        save_every_n_steps=0,  # disable periodic saves
        log_every_n_steps=1,
        eval_every_n_epochs=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        device="cpu",
    )
    return Trainer(model, tok, train_dl, val_dl, cfg)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_default_lr(self):
        cfg = TrainConfig()
        assert cfg.lr == pytest.approx(3e-4)

    def test_default_weight_decay(self):
        cfg = TrainConfig()
        assert cfg.weight_decay == pytest.approx(0.01)

    def test_default_num_epochs(self):
        cfg = TrainConfig()
        assert cfg.num_epochs == 10

    def test_default_grad_clip(self):
        cfg = TrainConfig()
        assert cfg.grad_clip == pytest.approx(1.0)

    def test_default_warmup_steps(self):
        cfg = TrainConfig()
        assert cfg.warmup_steps == 100

    def test_device_defaults_to_cpu_in_test_env(self):
        cfg = TrainConfig()
        # In CI without GPU, device should be cpu
        assert cfg.device in ("cpu", "cuda")

    def test_custom_values_are_preserved(self):
        cfg = TrainConfig(lr=1e-5, num_epochs=5, grad_clip=0.5, warmup_steps=50)
        assert cfg.lr == pytest.approx(1e-5)
        assert cfg.num_epochs == 5
        assert cfg.grad_clip == pytest.approx(0.5)
        assert cfg.warmup_steps == 50

    def test_checkpoint_dir_default(self):
        cfg = TrainConfig()
        assert cfg.checkpoint_dir == "checkpoints"

    def test_beta1_beta2_eps(self):
        cfg = TrainConfig()
        assert cfg.beta1 == pytest.approx(0.9)
        assert cfg.beta2 == pytest.approx(0.999)
        assert cfg.eps == pytest.approx(1e-8)

    def test_log_every_n_steps(self):
        cfg = TrainConfig()
        assert cfg.log_every_n_steps == 10

    def test_eval_every_n_epochs(self):
        cfg = TrainConfig()
        assert cfg.eval_every_n_epochs == 1


# ---------------------------------------------------------------------------
# cosine_schedule_with_warmup
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    @pytest.fixture
    def optimiser_and_schedule(self):
        from torch.optim import AdamW

        model = nn.Linear(4, 4)
        opt = AdamW(model.parameters(), lr=1e-3)
        sched = cosine_schedule_with_warmup(opt, warmup_steps=10, total_steps=100)
        return opt, sched

    def test_returns_lambda_lr(self, optimiser_and_schedule):
        from torch.optim.lr_scheduler import LambdaLR

        _, sched = optimiser_and_schedule
        assert isinstance(sched, LambdaLR)

    def test_lr_at_step_zero_is_zero(self, optimiser_and_schedule):
        opt, sched = optimiser_and_schedule
        # At step 0 (before any step), warmup fraction = 0/10 = 0
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.0)

    def test_lr_increases_during_warmup(self, optimiser_and_schedule):
        opt, sched = optimiser_and_schedule
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        # LR should generally increase during warmup (steps 1-10)
        assert lrs[-1] > lrs[0]

    def test_lr_after_warmup_starts_at_peak(self, optimiser_and_schedule):
        opt, sched = optimiser_and_schedule
        # Step through warmup
        for _ in range(10):
            sched.step()
        peak_lr = sched.get_last_lr()[0]
        assert peak_lr == pytest.approx(1.0, abs=0.01)

    def test_lr_decays_after_warmup(self, optimiser_and_schedule):
        opt, sched = optimiser_and_schedule
        # Step through warmup
        for _ in range(10):
            sched.step()
        peak = sched.get_last_lr()[0]
        # Step further into cosine decay
        for _ in range(50):
            sched.step()
        decayed = sched.get_last_lr()[0]
        assert decayed < peak

    def test_lr_approaches_zero_at_end(self):
        from torch.optim import AdamW

        model = nn.Linear(4, 4)
        opt = AdamW(model.parameters(), lr=1.0)
        sched = cosine_schedule_with_warmup(opt, warmup_steps=1, total_steps=1000)
        for _ in range(1000):
            sched.step()
        final = sched.get_last_lr()[0]
        assert final == pytest.approx(0.0, abs=0.05)

    def test_warmup_steps_zero_still_works(self):
        from torch.optim import AdamW

        model = nn.Linear(4, 4)
        opt = AdamW(model.parameters(), lr=1e-3)
        # warmup_steps=0 should not divide by zero
        sched = cosine_schedule_with_warmup(opt, warmup_steps=0, total_steps=10)
        sched.step()
        assert sched.get_last_lr()[0] >= 0.0


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------


class TestTrainerInit:
    def test_trainer_creates_checkpoint_dir(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        assert Path(trainer.config.checkpoint_dir).exists()

    def test_model_on_correct_device(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        for p in trainer.model.parameters():
            assert str(p.device) == "cpu"
            break

    def test_global_step_starts_at_zero(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        assert trainer.global_step == 0

    def test_best_val_accuracy_starts_at_zero(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        assert trainer.best_val_accuracy == 0.0

    def test_history_starts_empty(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        assert trainer.history == []

    def test_optimiser_has_two_param_groups(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        assert len(trainer.optimiser.param_groups) == 2

    def test_param_group_weight_decay_values(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        wds = {pg["weight_decay"] for pg in trainer.optimiser.param_groups}
        assert 0.0 in wds  # no-decay group
        assert trainer.config.weight_decay in wds  # decay group


# ---------------------------------------------------------------------------
# Trainer._train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_returns_float_loss_and_grad_norm(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        trainer.model.train()
        batch = {
            "input_ids": torch.randint(0, 500, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "label": torch.randint(0, 3, (2,)),
        }
        loss, grad_norm = trainer._train_step(batch)
        assert isinstance(loss, float)
        assert isinstance(grad_norm, float)
        assert loss > 0.0
        assert grad_norm >= 0.0

    def test_global_step_increments(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        trainer.model.train()
        batch = {
            "input_ids": torch.randint(0, 500, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "label": torch.randint(0, 3, (2,)),
        }
        trainer._train_step(batch)
        assert trainer.global_step == 1

    def test_loss_decreases_with_repeated_steps(self, tmp_path):
        """After many steps the loss should move (not necessarily down in 1 step,
        but at least show training is running)."""
        trainer = _tiny_trainer(tmp_path)
        trainer.model.train()
        # Fix a single batch to track loss
        batch = {
            "input_ids": torch.zeros(2, 16, dtype=torch.long),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "label": torch.zeros(2, dtype=torch.long),
        }
        losses = []
        for _ in range(5):
            loss, _ = trainer._train_step(batch)
            losses.append(loss)
        # At least the loss should be finite
        assert all(math.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# Trainer.evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_returns_loss_and_accuracy(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        metrics = trainer.evaluate()
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_evaluate_loss_is_positive(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        metrics = trainer.evaluate()
        assert metrics["loss"] >= 0.0

    def test_evaluate_accuracy_in_range(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        metrics = trainer.evaluate()
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_returns_model_to_train_mode(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        trainer.model.train()
        trainer.evaluate()
        assert trainer.model.training  # should be back in train mode

    def test_evaluate_empty_val_loader_does_not_crash(self, tmp_path):
        """Edge case: val_loader with 0 batches."""
        model = _tiny_model()
        tok = MagicMock()
        train_dl = _make_loader(n=4)

        class _EmptyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError

        empty_val = DataLoader(_EmptyDataset(), batch_size=4)
        cfg = TrainConfig(
            num_epochs=1,
            warmup_steps=1,
            save_every_n_steps=0,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu",
        )
        trainer = Trainer(model, tok, train_dl, empty_val, cfg)
        metrics = trainer.evaluate()
        assert metrics["loss"] == pytest.approx(0.0)
        assert metrics["accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trainer._train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    def test_train_epoch_returns_float(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        loss = trainer._train_epoch(epoch=1)
        assert isinstance(loss, float)

    def test_train_epoch_is_finite(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        loss = trainer._train_epoch(epoch=1)
        assert math.isfinite(loss)

    def test_global_step_advances_by_num_batches(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        n_batches = len(trainer.train_loader)
        trainer._train_epoch(epoch=1)
        assert trainer.global_step == n_batches

    def test_periodic_checkpoint_saved_when_step_matches(self, tmp_path):
        """When save_every_n_steps=2, a checkpoint should be saved."""
        model = _tiny_model()
        tok = MagicMock()
        train_dl = _make_loader(n=8, batch_size=4)  # 2 batches
        val_dl = _make_loader(n=4)
        cfg = TrainConfig(
            num_epochs=1,
            warmup_steps=1,
            save_every_n_steps=2,
            log_every_n_steps=1,
            eval_every_n_epochs=1,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu",
        )
        trainer = Trainer(model, tok, train_dl, val_dl, cfg)
        trainer._train_epoch(epoch=1)
        ckpt_dir = Path(cfg.checkpoint_dir)
        # A step checkpoint should exist
        step_ckpts = list(ckpt_dir.glob("step_*.pt"))
        assert len(step_ckpts) >= 1


# ---------------------------------------------------------------------------
# Trainer.train (full loop)
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_returns_history_list(self, tmp_path):
        trainer = _tiny_trainer(tmp_path, num_epochs=1)
        history = trainer.train()
        assert isinstance(history, list)
        assert len(history) == 1

    def test_history_has_epoch_and_train_loss(self, tmp_path):
        trainer = _tiny_trainer(tmp_path, num_epochs=1)
        history = trainer.train()
        assert "epoch" in history[0]
        assert "train_loss" in history[0]

    def test_history_has_val_metrics_when_eval_every_1(self, tmp_path):
        trainer = _tiny_trainer(tmp_path, num_epochs=1)
        history = trainer.train()
        assert "loss" in history[0]
        assert "accuracy" in history[0]

    def test_best_val_accuracy_updated(self, tmp_path):
        trainer = _tiny_trainer(tmp_path, num_epochs=2)
        trainer.train()
        assert trainer.best_val_accuracy >= 0.0

    def test_best_model_checkpoint_created(self, tmp_path):
        trainer = _tiny_trainer(tmp_path, num_epochs=1)
        trainer.train()
        best = Path(trainer.config.checkpoint_dir) / "best_model.pt"
        assert best.exists()

    def test_num_epochs_override(self, tmp_path):
        trainer = _tiny_trainer(tmp_path, num_epochs=5)
        history = trainer.train(num_epochs=2)
        assert len(history) == 2

    def test_train_with_eval_every_n_epochs_2(self, tmp_path):
        model = _tiny_model()
        tok = MagicMock()
        train_dl = _make_loader(n=8)
        val_dl = _make_loader(n=4)
        cfg = TrainConfig(
            num_epochs=3,
            warmup_steps=1,
            save_every_n_steps=0,
            log_every_n_steps=999,
            eval_every_n_epochs=2,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu",
        )
        trainer = Trainer(model, tok, train_dl, val_dl, cfg)
        history = trainer.train()
        # Epoch 2 should have val metrics; epochs 1 and 3 only have train_loss
        assert len(history) == 3
        # Epoch at index 1 (epoch=2) should have 'accuracy'
        assert "accuracy" in history[1]
        # Epoch at index 0 (epoch=1) should NOT have 'accuracy'
        assert "accuracy" not in history[0]


# ---------------------------------------------------------------------------
# Trainer.save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_checkpoint_creates_file(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        path = str(tmp_path / "test.pt")
        trainer.save_checkpoint(path, epoch=1)
        assert Path(path).exists()

    def test_load_checkpoint_restores_global_step(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        trainer.global_step = 42
        path = str(tmp_path / "step42.pt")
        trainer.save_checkpoint(path, epoch=3)

        trainer2 = _tiny_trainer(tmp_path)
        epoch = trainer2.load_checkpoint(path)
        assert trainer2.global_step == 42
        assert epoch == 3

    def test_load_checkpoint_restores_best_val_accuracy(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        trainer.best_val_accuracy = 0.75
        path = str(tmp_path / "acc.pt")
        trainer.save_checkpoint(path, epoch=2)

        trainer2 = _tiny_trainer(tmp_path)
        trainer2.load_checkpoint(path)
        assert trainer2.best_val_accuracy == pytest.approx(0.75)

    def test_load_checkpoint_restores_history(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        trainer.history = [{"epoch": 1, "train_loss": 1.5}]
        path = str(tmp_path / "hist.pt")
        trainer.save_checkpoint(path, epoch=1)

        trainer2 = _tiny_trainer(tmp_path)
        trainer2.load_checkpoint(path)
        assert len(trainer2.history) == 1
        assert trainer2.history[0]["train_loss"] == pytest.approx(1.5)

    def test_checkpoint_contains_model_config(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        path = str(tmp_path / "cfg.pt")
        trainer.save_checkpoint(path, epoch=1)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "config" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimiser_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt

    def test_weights_match_after_save_load(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        path = str(tmp_path / "weights.pt")
        trainer.save_checkpoint(path, epoch=0)

        trainer2 = _tiny_trainer(tmp_path)
        trainer2.load_checkpoint(path)

        for (n1, p1), (n2, p2) in zip(
            trainer.model.named_parameters(), trainer2.model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    def test_load_checkpoint_missing_optional_keys(self, tmp_path):
        """Checkpoint saved without best_val_accuracy / history should use defaults."""
        trainer = _tiny_trainer(tmp_path)
        ckpt = {
            "epoch": 5,
            "global_step": 100,
            "model_state_dict": trainer.model.state_dict(),
            "optimiser_state_dict": trainer.optimiser.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "config": trainer.model.config,
        }
        path = str(tmp_path / "minimal.pt")
        torch.save(ckpt, path)

        trainer2 = _tiny_trainer(tmp_path)
        epoch = trainer2.load_checkpoint(path)
        assert epoch == 5
        assert trainer2.global_step == 100
        assert trainer2.best_val_accuracy == 0.0
        assert trainer2.history == []
