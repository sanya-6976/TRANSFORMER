import torch 

from tasks.model.feed_forward import FeedForward

torch.manual_seed(42)

# Test multiple configurations
configurations = [
    (64, 4, 2, 5),
    (128, 4, 2, 5),
    (256, 4, 4, 8)
]
for embedding_dim, expansion, batch_size, seq_len in configurations:

    print("\n----------------------------")
    print("Embedding dimension:", embedding_dim)

    model = FeedForward(
        embedding_dim=embedding_dim,
        expansion_factor=expansion,
        dropout=0.0
    )

    x = torch.randn(batch_size, seq_len, embedding_dim)

    output = model(x)

    hidden_dim = embedding_dim * expansion

    expected_params = (
        embedding_dim * hidden_dim + hidden_dim
        + hidden_dim * embedding_dim + embedding_dim
    )
    actual_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Input shape:", x.shape)
    print("Hidden dimension:", hidden_dim)
    print("Output shape:", output.shape)
    print("Expected parameters:", expected_params)
    print("Actual parameters:", actual_params)
    assert output.shape == x.shape
    assert actual_params == expected_params

    loss = output.mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name

    print("Shape test: PASSED")
    print("Parameter test: PASSED")
    print("Gradient test: PASSED")

print("\nALL FFN TESTS PASSED!")