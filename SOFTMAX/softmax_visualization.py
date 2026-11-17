
import torch
import matplotlib.pyplot as plt

tokens = ["sleeping", "running", "eating"]

logits = torch.tensor([2.0, 1.0, 0.1])

probabilities = torch.softmax(logits, dim=0)

plt.bar(tokens, probabilities.numpy())

plt.title("GPT Next Token Probabilities")
plt.xlabel("Possible Next Tokens")
plt.ylabel("Probability")

plt.ylim(0, 1)

for i, probability in enumerate(probabilities):
    plt.text(
        i,
        probability.item() + 0.02,
        f"{probability.item():.3f}",
        ha="center"
    )

plt.show()
