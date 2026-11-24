
import torch

from tasks.model.gpt_model import GPTModel


# ==================================================
# 1. CREATE GPT MODEL
# ==================================================

def create_model(tie_weights=True):

    model = GPTModel(
        vocab_size=10000,
        max_seq_len=64,
        embedding_dim=128,
        num_heads=4,
        num_layers=6,
        expansion_factor=4,
        dropout=0.0,
        tie_weights=tie_weights
    )

    return model


# ==================================================
# 2. TEST FORWARD PASS
# ==================================================

def test_forward_pass():

    print("\n1. Testing forward pass...")

    model = create_model()

    token_ids = torch.randint(
        0,
        10000,
        (2, 5)
    )

    logits = model(token_ids)

    assert logits.shape == (2, 5, 10000)

    assert torch.isfinite(logits).all()

    print("Input shape:", token_ids.shape)
    print("Output shape:", logits.shape)

    print("Forward pass passed!")


# ==================================================
# 3. TEST CAUSAL BEHAVIOR
# ==================================================

def test_causal_behavior():

    print("\n2. Testing causal behavior...")

    model = create_model()

    model.eval()

    original_tokens = torch.tensor(
        [[10, 20, 30, 40, 50]]
    )

    changed_tokens = torch.tensor(
        [[10, 20, 30, 40, 99]]
    )

    with torch.no_grad():

        original_logits = model(original_tokens)

        changed_logits = model(changed_tokens)

    # Earlier positions must remain unchanged.

    assert torch.allclose(
        original_logits[:, :4, :],
        changed_logits[:, :4, :],
        atol=1e-6
    )

    # The changed final token should affect its own output.

    assert not torch.allclose(
        original_logits[:, 4, :],
        changed_logits[:, 4, :],
        atol=1e-6
    )

    print("Earlier position logits remained unchanged!")

    print("Final position logits changed!")

    print("Causal behavior passed!")


# ==================================================
# 4. TEST WEIGHT TYING
# ==================================================

def test_weight_tying():

    print("\n3. Testing tied and untied weights...")

    # Create tied model.

    tied_model = create_model(
        tie_weights=True
    )

    input_weight = (
        tied_model.input_embedding
        .token_embedding.embedding.weight
    )

    output_weight = (
        tied_model.language_model_head
        .projection.weight
    )

    assert input_weight is output_weight

    # Create untied model.

    untied_model = create_model(
        tie_weights=False
    )

    input_weight_untied = (
        untied_model.input_embedding
        .token_embedding.embedding.weight
    )

    output_weight_untied = (
        untied_model.language_model_head
        .projection.weight
    )

    assert input_weight_untied is not output_weight_untied

    # Compare parameter counts.

    tied_count = sum(
        parameter.numel()
        for parameter in tied_model.parameters()
        if parameter.requires_grad
    )

    untied_count = sum(
        parameter.numel()
        for parameter in untied_model.parameters()
        if parameter.requires_grad
    )

    assert tied_count == 2488080

    assert untied_count == 3768080

    print("Tied model parameters:", tied_count)

    print("Untied model parameters:", untied_count)

    print("Weight tying configuration passed!")


# ==================================================
# 5. TEST CONTEXT LENGTH LIMIT
# ==================================================

def test_context_limit():

    print("\n4. Testing context length limit...")

    model = create_model()

    # Test maximum allowed sequence length.

    valid_tokens = torch.randint(
        0,
        10000,
        (2, 64)
    )

    valid_logits = model(valid_tokens)

    assert valid_logits.shape == (2, 64, 10000)

    print("Maximum context length accepted!")

    # Test sequence longer than allowed.

    invalid_tokens = torch.randint(
        0,
        10000,
        (2, 65)
    )

    try:

        model(invalid_tokens)

    except ValueError:

        print("Oversized sequence correctly rejected!")

    else:

        raise AssertionError(
            "Model accepted an oversized sequence!"
        )

    print("Context length test passed!")


# ==================================================
# 6. TEST GRADIENT FLOW
# ==================================================

def test_gradient_flow():

    print("\n5. Testing gradient flow...")

    model = create_model()

    token_ids = torch.randint(
        0,
        10000,
        (2, 5)
    )

    targets = torch.randint(
        0,
        10000,
        (2, 5)
    )

    # Forward pass.

    logits = model(token_ids)

    # Calculate cross-entropy loss.

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 10000),
        targets.reshape(-1)
    )

    assert torch.isfinite(loss)

    # Backpropagation.

    loss.backward()

    parameters_with_gradients = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            assert parameter.grad is not None, (
                f"Missing gradient: {name}"
            )

            assert torch.isfinite(
                parameter.grad
            ).all(), (
                f"Non-finite gradient: {name}"
            )

            parameters_with_gradients += 1

    print("Training loss:", loss.item())

    print(
        "Parameters with gradients:",
        parameters_with_gradients
    )

    print("Gradient flow passed!")


# ==================================================
# 7. RUN ALL INTEGRATION TESTS
# ==================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    test_forward_pass()

    test_causal_behavior()

    test_weight_tying()

    test_context_limit()

    test_gradient_flow()

    print("\nAll Complete GPT Integration Tests Passed!")
