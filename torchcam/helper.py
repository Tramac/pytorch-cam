import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

__all__ = ['open_image', 'show_image', 'image2batch', 'int2tensor']


def _resize_image(img, size=None):
    if size is not None or (hasattr(img, 'shape') and len(img.shape) == 1):
        if size is None:
            # make guess for 1-dim tensors
            h = int(math.sqrt(img.shape[0]))
            w = int(img.shape[0] / h)
            size = h, w
        img = np.reshape(img, size)
    return img


def open_image(path, resize=None, resample=1, convert_mode=None):
    img = Image.open(path)
    if resize is not None:
        img = img.resize(resize, resample)
    if convert_mode is not None:
        img = img.convert(convert_mode)
    return img


def show_image(img, size=None, alpha=None, cmap=None, img2=None, size2=None, alpha2=None, cmap2=None, ax=None):
    img = _resize_image(img, size)
    img2 = _resize_image(img2, size2)

    (ax or plt).imshow(img, alpha=alpha, cmap=cmap)

    if img2 is not None:
        (ax or plt).imshow(img2, alpha=alpha2, cmap=cmap2)

    return ax or plt.show()


def get_image_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return transform


def image2batch(image):
    image_transform = get_image_transform()
    if image_transform:
        input_x = image_transform(image)
    else:
        input_x = transforms.ToTensor()(image)
    input_x = input_x.unsqueeze(0)

    return input_x


def int2tensor(val):
    return torch.LongTensor([val])
