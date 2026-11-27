
import torch

from tasks.model.generation import sample_next_token


def test_output_shape_and_dtype():
    logits = torch.tensor([
        [2.0, 1.0, 0.5, 0.1],
        [0.1, 0.5, 1.0, 2.0]
    ])

    tokens = sample_next_token(logits)

    assert tokens.shape == (2, 1)
    assert tokens.dtype == torch.long

    print("Output shape and dtype test passed!")


def test_token_range():
    logits = torch.randn(4, 100)

    tokens = sample_next_token(
        logits,
        temperature=0.8,
        top_k=20,
        top_p=0.9
    )

    assert torch.all(tokens >= 0)
    assert torch.all(tokens < 100)

    print("Token range test passed!")


def test_top_k_one():
    logits = torch.tensor([
        [1.0, 5.0, 2.0, 3.0]
    ])

    for _ in range(10):
        token = sample_next_token(
            logits,
            top_k=1
        )

        assert token.item() == 1

    print("Top-K = 1 test passed!")


def test_top_p_one():
    logits = torch.tensor([
        [2.0, 1.0, 0.5]
    ])

    torch.manual_seed(42)

    first = sample_next_token(
        logits,
        top_p=1.0
    )

    torch.manual_seed(42)

    second = sample_next_token(
        logits,
        top_p=1.0
    )

    assert torch.equal(first, second)

    print("Top-P = 1 test passed!")


def test_combined_sampling():
    logits = torch.randn(4, 100)

    tokens = sample_next_token(
        logits,
        temperature=0.8,
        top_k=20,
        top_p=0.9
    )

    assert tokens.shape == (4, 1)
    assert torch.all(tokens < 100)

    print("Combined sampling test passed!")


def test_invalid_temperature():
    logits = torch.randn(1, 10)

    try:
        sample_next_token(logits, temperature=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    print("Invalid temperature test passed!")


def test_invalid_top_k():
    logits = torch.randn(1, 10)

    try:
        sample_next_token(logits, top_k=11)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    print("Invalid Top-K test passed!")


def test_invalid_top_p():
    logits = torch.randn(1, 10)

    try:
        sample_next_token(logits, top_p=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    print("Invalid Top-P test passed!")


def test_non_finite_logits():
    logits = torch.tensor([
        [1.0, float("nan"), 2.0]
    ])

    try:
        sample_next_token(logits)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    print("Non-finite logits test passed!")


def run_all_tests():
    test_output_shape_and_dtype()
    test_token_range()
    test_top_k_one()
    test_top_p_one()
    test_combined_sampling()
    test_invalid_temperature()
    test_invalid_top_k()
    test_invalid_top_p()
    test_non_finite_logits()

    print("\nAll sampling tests passed!")


if __name__ == "__main__":
    run_all_tests()
