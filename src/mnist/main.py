import torchvision


def main():
    torchvision.datasets.MNIST(
        root="data",
        download=True,
    )


if __name__ == "__main__":
    main()
