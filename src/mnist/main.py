import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import random_split
from torchvision.transforms import transforms

NUM_IMAGES_TRAIN = 50_000
NUM_IMAGES_VALIDATION = 10_000


def load(load_as_tensor: bool = True):
    mnist = torchvision.datasets.MNIST(
        root="data",
        download=True,
        transform=transforms.ToTensor() if load_as_tensor else None,
    )
    return mnist


def load_as_tensor():
    """
    Dimensions: (N, 1, 28, 28)

    N: number of images
    1: number of channels (grayscale). For RGB images, this would be 3.
    28, 28: height and width of each image
    """
    return load(load_as_tensor=True)


def main_pil():
    mnist = load(load_as_tensor=False)
    for image, label in mnist:
        image: Image
        image.show()
        print(label)
        break


def main_tensor():
    mnist = load_as_tensor()
    NUM_IMAGES = len(mnist)
    print(NUM_IMAGES)

    print(type(mnist))
    for image, label in mnist:
        image: torch.Tensor
        print(image.shape, image.dtype)
        print(image[0])
        break

    train_data, validation_data = random_split(
        mnist, [NUM_IMAGES_TRAIN, NUM_IMAGES_VALIDATION]
    )


if __name__ == "__main__":
    main_tensor()
