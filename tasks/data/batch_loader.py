import torch 

def get_batch(
        data, 
        batch_size , 
        block_size , 
        device="cpu"
):
    """
    Randomly sample input and next-token target batches.

    Args:
        data: 1D tensor containing token IDs.
        batch_size: Number of sequences in one batch.
        block_size: Number of tokens per sequence.
        device: Device where the batches should be placed.

    Returns:
        x: Input token IDs, shape (batch_size, block_size).
        y: Target token IDs, shape (batch_size, block_size).
    """ 

    # validate the input data 
    if not isinstance(data , torch.Tensor):
        raise TypeError("data must be a PyTorch tensor")

    if data.ndim != 1:
        raise ValueError("data must be a 1D tensor")

    if data.dtype != torch.long:
        raise TypeError("data must have dtype torch.long")

    if batch_size <= 0 or block_size <= 0:
        raise ValueError(
            "batch_size and block_size must be positive"
        )

    