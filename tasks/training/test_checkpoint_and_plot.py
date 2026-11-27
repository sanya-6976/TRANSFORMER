
import tempfile
from pathlib import Path

import torch
from torch import nn

from tasks.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from tasks.training.plot_loss import plot_loss


def test_checkpoint_save_and_load():
    print("\n1. Testing checkpoint save and load...")

    torch.manual_seed(42)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    # Perform one update so the optimizer has state to save.
    sample_input = torch.randn(2, 4)
    loss = model(sample_input).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Remember the trained weights.
    expected_weights = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    history = [
        {
            "step": 0,
            "train_loss": 4.5,
            "val_loss": 4.8
        },
        {
            "step": 10,
            "train_loss": 3.9,
            "val_loss": 4.2
        }
    ]

    config = {
        "batch_size": 4,
        "block_size": 64
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir) / "test_checkpoint.pt"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=10,
            history=history,
            path=checkpoint_path,
            config=config
        )

        assert checkpoint_path.exists()

        # Change the weights to prove loading restores them.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)

        restored = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            device="cpu"
        )

        for name, parameter in model.named_parameters():
            assert torch.equal(
                parameter,
                expected_weights[name]
            )

        assert restored["step"] == 10
        assert restored["history"] == history
        assert restored["config"] == config

    print("Checkpoint save/load passed!")


def test_loss_graph():
    print("\n2. Testing loss graph generation...")

    history = [
        {
            "step": 0,
            "train_loss": 5.0,
            "val_loss": 5.2
        },
        {
            "step": 100,
            "train_loss": 4.0,
            "val_loss": 4.3
        },
        {
            "step": 200,
            "train_loss": 3.2,
            "val_loss": 3.7
        }
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        graph_path = (
            Path(temp_dir) / "test_training_loss.png"
        )

        plot_loss(
            history=history,
            save_path=graph_path
        )

        assert graph_path.exists()
        assert graph_path.stat().st_size > 0

    print("Loss graph generation passed!")


if __name__ == "__main__":
    test_checkpoint_save_and_load()
    test_loss_graph()

    print("\nAll Phase 8.7 tests passed!")
