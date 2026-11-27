
from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    step,
    history,
    path,
    config=None
):
    """
    Save model, optimizer, training progress and configuration.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config
    }

    # Save to a temporary file first.
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path
    )

    # Replace the destination after saving succeeds.
    temporary_path.replace(path)

    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path,
    model,
    optimizer,
    device="cpu"
):
    """
    Restore model and optimizer states from a checkpoint.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    print(f"Checkpoint loaded: {path}")
    print(
        f"Resuming from step: {checkpoint['step']}"
    )

    return {
        "step": checkpoint["step"],
        "history": checkpoint["history"],
        "config": checkpoint["config"]
    }
