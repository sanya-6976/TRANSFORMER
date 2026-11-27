
import torch
import torch.nn as nn

from tasks.model.generation import generate


class DummyGPT(nn.Module):

    def __init__(self, vocab_size=20):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_lengths = []

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        self.context_lengths.append(sequence_length)

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size
        )

        # Token ID 7 always has the highest logit
        logits[:, :, 7] = 10.0

        return logits


def test_output_shape():

    model = DummyGPT()

    prompt = torch.tensor([[1, 2, 3]])

    output = generate(
        model,
        prompt,
        max_new_tokens=5,
        block_size=8
    )

    assert output.shape == (1, 8)

    print("Output shape test passed!")


def test_greedy_generation():

    model = DummyGPT()

    prompt = torch.tensor([[1, 2]])

    output = generate(
        model,
        prompt,
        max_new_tokens=3,
        block_size=8,
        strategy="greedy"
    )

    assert torch.equal(
        output[0, 2:],
        torch.tensor([7, 7, 7])
    )

    print("Greedy generation test passed!")


def test_sampling_strategies():

    model = DummyGPT()

    prompt = torch.tensor([[1, 2]])

    strategies = [
        ("temperature", {}),
        ("top_k", {"top_k": 5}),
        ("top_p", {"top_p": 0.9}),
        ("combined", {"top_k": 5, "top_p": 0.9})
    ]

    for strategy, options in strategies:

        output = generate(
            model,
            prompt,
            max_new_tokens=4,
            block_size=8,
            strategy=strategy,
            temperature=0.8,
            **options
        )

        assert output.shape == (1, 6)
        assert torch.all(output >= 0)
        assert torch.all(output < model.vocab_size)

    print("All sampling strategies passed!")


def test_context_cropping():

    model = DummyGPT()

    prompt = torch.tensor([[1, 2, 3, 4]])

    output = generate(
        model,
        prompt,
        max_new_tokens=5,
        block_size=4
    )

    assert output.shape == (1, 9)
    assert max(model.context_lengths) <= 4
    assert model.context_lengths[-1] == 4

    print("Context cropping test passed!")


def test_model_mode_restoration():

    model = DummyGPT()
    model.train()

    prompt = torch.tensor([[1, 2]])

    generate(
        model,
        prompt,
        max_new_tokens=2,
        block_size=8
    )

    assert model.training is True

    model.eval()

    generate(
        model,
        prompt,
        max_new_tokens=2,
        block_size=8
    )

    assert model.training is False

    print("Model mode restoration test passed!")


def test_zero_new_tokens():

    model = DummyGPT()

    prompt = torch.tensor([[1, 2, 3]])

    output = generate(
        model,
        prompt,
        max_new_tokens=0,
        block_size=8
    )

    assert torch.equal(output, prompt)

    print("Zero-token generation test passed!")


def test_invalid_initial_context():

    model = DummyGPT()

    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    try:
        generate(
            model,
            prompt,
            max_new_tokens=2,
            block_size=4
        )
        raise AssertionError("Expected ValueError")

    except ValueError:
        pass

    print("Invalid initial context test passed!")


def run_all_tests():

    test_output_shape()
    test_greedy_generation()
    test_sampling_strategies()
    test_context_cropping()
    test_model_mode_restoration()
    test_zero_new_tokens()
    test_invalid_initial_context()

    print("\nAll GPT generation tests passed!")


if __name__ == "__main__":
    run_all_tests()
