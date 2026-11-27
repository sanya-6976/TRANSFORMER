
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
