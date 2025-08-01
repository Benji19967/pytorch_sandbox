import time

import torch


def main():
    device = torch.device("mps")

    X = torch.rand(size=(20000, 10000))
    Y = torch.rand(size=(10000, 20000))

    # CPU
    s = time.time()
    print(X @ Y)
    print(time.time() - s)

    # GPU
    X = X.to(device)
    Y = Y.to(device)

    s = time.time()
    print(X @ Y)
    print(time.time() - s)


if __name__ == "__main__":
    main()
