
import torch

from tasks.data.batch_loader import get_batch
from tasks.model.language_model_loss import LanguageModelLoss
from tasks.training.checkpoint import save_checkpoint
from tasks.training.plot_loss import plot_loss

# ==================================================
# 1. ESTIMATE TRAINING / VALIDATION LOSS
# ==================================================

@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    eval_iters,
    device="cpu"
):
    # Remember the model's current mode.
    was_training = model.training

    # Switch off training-specific behavior.
    model.eval()

    loss_function = LanguageModelLoss()

    results = {}

    datasets = {
        "train": train_data,
        "val": val_data
    }

    for split, data in datasets.items():

        losses = []

        for _ in range(eval_iters):

            # Sample a batch.
            x, y = get_batch(
                data=data,
                batch_size=batch_size,
                block_size=block_size,
                device=device
            )

            # Forward pass.
            logits = model(x)

            # Calculate loss.
            loss = loss_function(logits, y)

            losses.append(loss.item())

        # Average loss across evaluation batches.
        results[split] = sum(losses) / len(losses)

    # Restore the previous model mode.
    model.train(was_training)

    return results


# ==================================================
# 2. MAIN TRAINING LOOP
# ==================================================

def train_model(
    model,
    train_data,
    val_data,
    optimizer,
    batch_size=4,
    block_size=64,
    max_steps=1000,
    eval_interval=100,
    eval_iters=20,
    device="cpu",
    max_grad_norm=1.0,
    latest_checkpoint_path="checkpoints/latest_checkpoint.pt",
    best_checkpoint_path="checkpoints/best_checkpoint.pt",
    loss_plot_path="outputs/training_loss.png"
):

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if eval_interval <= 0 or eval_iters <= 0:
        raise ValueError(
            "eval_interval and eval_iters must be positive"
        )

    if batch_size <= 0 or block_size <= 0:
        raise ValueError(
            "batch_size and block_size must be positive"
        )

    if max_grad_norm <= 0:
        raise ValueError(
            "max_grad_norm must be positive"
        )

    model.train()

    loss_function = LanguageModelLoss()

    history = []

    # Track the lowest validation loss.
    best_val_loss = float("inf")

    # Store training configuration.
    config = {
        "batch_size": batch_size,
        "block_size": block_size,
        "max_steps": max_steps,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters,
        "max_grad_norm": max_grad_norm
    }

    # ==================================================
    # 3. REPEAT TRAINING STEPS
    # ==================================================

    for step in range(max_steps):

        # 1. Get a random training batch.
        x, y = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device
        )

        # 2. Clear previous gradients.
        optimizer.zero_grad(set_to_none=True)

        # 3. Forward pass.
        logits = model(x)

        # 4. Calculate training loss.
        loss = loss_function(logits, y)

        # 5. Backpropagation.
        loss.backward()

        # 6. Prevent excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )

        # 7. Update model parameters.
        optimizer.step()

        # ==================================================
        # 4. VALIDATION AND CHECKPOINT SAVING
        # ==================================================

        if step % eval_interval == 0 or step == max_steps - 1:

            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
                device=device
            )

            record = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"]
            }

            history.append(record)

            print(
                f"Step {step:5d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Validation Loss: {losses['val']:.4f}"
            )

            # ------------------------------------------
            # SAVE LATEST CHECKPOINT
            # ------------------------------------------

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                history=history,
                path=latest_checkpoint_path,
                config=config
            )

            # ------------------------------------------
            # SAVE BEST CHECKPOINT
            # ------------------------------------------

            if losses["val"] < best_val_loss:

                best_val_loss = losses["val"]

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    history=history,
                    path=best_checkpoint_path,
                    config=config
                )

                print(
                    f"New best model! "
                    f"Validation Loss: {best_val_loss:.4f}"
                )

            # ------------------------------------------
            # UPDATE LOSS GRAPH
            # ------------------------------------------

            plot_loss(
                history=history,
                save_path=loss_plot_path
            )

    print("\nTraining completed!")

    print(
        f"Best validation loss: {best_val_loss:.4f}"
    )

    print(
        "Latest checkpoint:",
        latest_checkpoint_path
    )

    print(
        "Best checkpoint:",
        best_checkpoint_path
    )

    print(
        "Loss graph:",
        loss_plot_path
    )

    return history
