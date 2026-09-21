import torch


def dropout(x, p=0.5, training=True):

    # During testing dropout is not applied
    if not training:
        return x

    if p < 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")

    # Random numbers between 0 and 1
    random_values = torch.rand_like(x)

    # Create mask
    mask = (random_values > p).float()

    # Apply dropout
    output = x * mask

    # Inverted dropout scaling
    output = output / (1 - p)

    return output


# Example
x = torch.tensor([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.]
])

print("Original:")
print(x)

print("\nAfter Dropout:")
print(dropout(x, p=0.5))