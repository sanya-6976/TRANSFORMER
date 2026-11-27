
import torch


@torch.no_grad()
def generate_greedy(
    model,
    input_ids,
    max_new_tokens,
    block_size
):
    """
    Generate tokens using greedy decoding.

    Args:
        model: Our GPT model.
        input_ids: Starting token IDs, shape (B, T).
        max_new_tokens: Number of tokens to generate.
        block_size: Maximum context length.

    Returns:
        Generated token IDs.
    """

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape (B, T)"
        )

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens cannot be negative"
        )

    if block_size <= 0:
        raise ValueError(
            "block_size must be positive"
        )

    if input_ids.shape[1] == 0:
        raise ValueError(
            "input_ids cannot be empty"
        )

    if input_ids.shape[1] > block_size:
        raise ValueError(
            "Initial sequence exceeds block_size"
        )

    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):

            # Keep only the latest context tokens.
            context = input_ids[:, -block_size:]

            # Get model predictions.
            logits = model(context)

            # Select the final position's logits.
            next_token_logits = logits[:, -1, :]

            # Select the highest-scoring token.
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            # Append the predicted token.
            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1
            )

    finally:
        model.train(was_training)

    return input_ids


def sample_temperature(logits, temperature=1.0):
    """
    Select a token using temperature sampling.

    Args:
        logits: Raw model logits, shape (B, V).
        temperature: Positive temperature value.

    Returns:
        Sampled token IDs, shape (B, 1).
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape (B, V)"
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero"
        )

    # Adjust logits using temperature.
    scaled_logits = logits / temperature

    # Convert adjusted logits into probabilities.
    probabilities = torch.softmax(
        scaled_logits,
        dim=-1
    )

    # Randomly sample one token.
    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token


def sample_top_k(
    logits,
    k=10,
    temperature=1.0
):
    """
    Sample a token from the top K candidates.

    Args:
        logits: Raw logits, shape (B, V).
        k: Number of highest-scoring tokens to retain.
        temperature: Positive sampling temperature.

    Returns:
        Sampled token IDs, shape (B, 1).
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape (B, V)"
        )

    vocab_size = logits.shape[-1]

    if not 1 <= k <= vocab_size:
        raise ValueError(
            "k must be between 1 and vocabulary size"
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero"
        )

    # Apply temperature.
    scaled_logits = logits / temperature

    # Find the K highest-scoring logits.
    top_values, _ = torch.topk(
        scaled_logits,
        k=k,
        dim=-1
    )

    # Get the smallest value among the top K.
    threshold = top_values[:, -1:]

    # Remove all tokens outside the top K.
    filtered_logits = scaled_logits.masked_fill(
        scaled_logits < threshold,
        float("-inf")
    )

    # Convert filtered logits into probabilities.
    probabilities = torch.softmax(
        filtered_logits,
        dim=-1
    )

    # Randomly select one remaining token.
    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token


def sample_top_p(
    logits,
    p=0.9,
    temperature=1.0
):
    """
    Sample a token using nucleus (Top-P) sampling.

    Args:
        logits: Raw logits, shape (B, V).
        p: Cumulative probability threshold.
        temperature: Positive sampling temperature.

    Returns:
        Sampled token IDs, shape (B, 1).
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape (B, V)"
        )

    if not 0 < p <= 1:
        raise ValueError(
            "p must be greater than 0 and at most 1"
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero"
        )

    # Apply temperature.
    scaled_logits = logits / temperature

    # Convert logits into probabilities.
    probabilities = torch.softmax(
        scaled_logits,
        dim=-1
    )

    # Sort probabilities from highest to lowest.
    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=-1
    )

    # Calculate cumulative probabilities.
    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    # Mark tokens whose preceding cumulative
    # probability has already reached the threshold.
    sorted_indices_to_remove = (
        cumulative_probs - sorted_probs
    ) >= p

    # Always retain the highest-probability token.
    sorted_indices_to_remove[:, 0] = False

    # Map sorted positions back to vocabulary indices.
    indices_to_remove = torch.zeros_like(
        sorted_indices_to_remove
    )

    indices_to_remove.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )

    # Remove tokens outside the nucleus.
    filtered_logits = scaled_logits.masked_fill(
        indices_to_remove,
        float("-inf")
    )

    # Calculate probabilities again.
    filtered_probs = torch.softmax(
        filtered_logits,
        dim=-1
    )

    # Sample one token.
    next_token = torch.multinomial(
        filtered_probs,
        num_samples=1
    )

    return next_token

def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=1.0
):
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1")

    vocab_size = logits.shape[-1]

    if top_k is not None:
        if not 1 <= top_k <= vocab_size:
            raise ValueError("Invalid top_k")

    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")

    filtered_logits = logits / temperature

    # Top-K filtering
    if top_k is not None and top_k < vocab_size:
        top_values, top_indices = torch.topk(
            filtered_logits,
            k=top_k,
            dim=-1
        )

        top_k_logits = torch.full_like(
            filtered_logits,
            float("-inf")
        )

        top_k_logits.scatter_(
            dim=-1,
            index=top_indices,
            src=top_values
        )

        filtered_logits = top_k_logits

    # Top-P filtering
    if top_p < 1.0:
        probabilities = torch.softmax(
            filtered_logits,
            dim=-1
        )

        sorted_probs, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=-1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        sorted_remove = (
            cumulative_probs - sorted_probs
        ) >= top_p

        sorted_remove[:, 0] = False

        remove_indices = torch.zeros_like(
            sorted_remove
        )

        remove_indices.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_remove
        )

        filtered_logits = filtered_logits.masked_fill(
            remove_indices,
            float("-inf")
        )

    probabilities = torch.softmax(
        filtered_logits,
        dim=-1
    )

    return torch.multinomial(
        probabilities,
        num_samples=1
    )


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    block_size,
    strategy="greedy",
    temperature=1.0,
    top_k=None,
    top_p=0.9
):
    """
    Generate tokens autoregressively using a GPT model.
    """

    # Step 1: Validate input
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (B, T)")

    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must have dtype torch.long")

    if input_ids.shape[1] == 0:
        raise ValueError("input_ids cannot be empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if input_ids.shape[1] > block_size:
        raise ValueError("Initial sequence exceeds block_size")

    valid_strategies = {
        "greedy",
        "temperature",
        "top_k",
        "top_p",
        "combined"
    }

    if strategy not in valid_strategies:
        raise ValueError("Unknown generation strategy")

    if strategy != "greedy" and temperature <= 0:
        raise ValueError("Temperature must be greater than zero")

    if strategy in {"top_p", "combined"}:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

    # Step 2: Preserve the original prompt
    input_ids = input_ids.clone()

    # Step 3: Remember the original model mode
    was_training = model.training
    model.eval()

    try:

        # Step 4: Generate tokens one by one
        for _ in range(max_new_tokens):

            # Keep only the latest context window
            context = input_ids[:, -block_size:]

            # GPT forward pass
            logits = model(context)

            # Validate model output
            if logits.ndim != 3:
                raise ValueError("Model must return logits of shape (B, T, V)")

            if logits.shape[0] != input_ids.shape[0]:
                raise ValueError("Model output batch size is incorrect")

            if logits.shape[1] != context.shape[1]:
                raise ValueError("Model output sequence length is incorrect")

            # Select logits from the final position
            next_token_logits = logits[:, -1, :]

            if not torch.isfinite(next_token_logits).all():
                raise ValueError("Model produced non-finite logits")

            # Step 5: Select the next token
            if strategy == "greedy":

                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True
                )

            elif strategy == "temperature":

                next_token = sample_next_token(
                    next_token_logits,
                    temperature=temperature
                )

            elif strategy == "top_k":

                if top_k is None:
                    raise ValueError("top_k is required for top_k strategy")

                next_token = sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k
                )

            elif strategy == "top_p":

                next_token = sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_p=top_p
                )

            else:

                next_token = sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

            # Step 6: Append the new token
            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1
            )

    finally:
        # Restore the original model mode
        model.train(was_training)

    return input_ids
