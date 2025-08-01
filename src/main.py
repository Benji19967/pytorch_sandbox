import torch

torch.device("mps")


def main():
    print(torch.__version__)

    # zero-dimensional tensor
    scalar = torch.tensor(5)
    print(scalar, scalar.ndim)

    vector = torch.tensor([3, 2, 5])
    print(vector, vector.ndim, vector.shape)

    MATRIX = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )
    print(MATRIX, MATRIX.ndim, MATRIX.shape)

    TENSOR = torch.tensor(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        ]
    )
    print(TENSOR, TENSOR.ndim, TENSOR.shape)

    # Creating random tensors
    X = torch.rand(size=(5, 3))
    print(X, X.shape, X.dtype)

    Y = torch.rand(size=(224, 224, 3))
    print(Y.shape, Y.dtype)


if __name__ == "__main__":
    main()
