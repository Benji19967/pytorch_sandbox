import torchvision


def main():
    mnist = torchvision.datasets.MNIST(
        root="data",
        download=True,
    )

    for image, label in mnist:
        image.show()
        print(label)
        break


if __name__ == "__main__":
    main()
