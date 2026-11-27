
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss(
    history,
    save_path="outputs/training_loss.png"
):
    if not history:
        raise ValueError("Training history is empty")

    steps = [
        item["step"]
        for item in history
    ]

    train_losses = [
        item["train_loss"]
        for item in history
    ]

    val_losses = [
        item["val_loss"]
        for item in history
    ]

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        steps,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("GPT Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.close()

    print(f"Loss graph saved: {save_path}")
