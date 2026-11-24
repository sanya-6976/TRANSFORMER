import torch  

from tasks.data.batch_loader import  get_batch 

def test_batch_shapes():

    data = torch.arange(1000 , dtype=torch.long)

    x , y = get_batch(
        data=data,
        batch_size = 4 , 
        block_size = 64
    )

    assert x.shape == (4, 64)
    assert y.shape == (4, 64)

    print("Batch shape test passed!")

def test_next_token_alignment():

    data = torch.arange(1000 ,dtype=torch.long)

    x, y = get_batch(
        data=data,
        batch_size=4,
        block_size=64
    )

    # Every target must be the next token
    # after its corresponding input token.
    assert torch.equal(x[:, 1:], y[:, :-1])

    print("Next-token alignment test passed!")

def test_invalid_dataset():

    data = torch.tensor([10, 20, 30], dtype=torch.long)

    try:
        get_batch(
            data=data,
            batch_size=2,
            block_size=3
        )

    except ValueError:
        print("Invalid dataset test passed!")

    else:
        raise AssertionError(
            "Expected ValueError for insufficient data"
        )


def test_all():

    test_batch_shapes()
    test_next_token_alignment()
    test_invalid_dataset()

    print("\nAll batch loader tests passed!")


if __name__ == "__main__":
    test_all()
