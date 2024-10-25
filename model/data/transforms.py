from functools import lru_cache
from data.functions import get_image, get_urls
import cv2
import numpy as np


def transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)
    edges = np.expand_dims(edges, axis=2)
    img = np.expand_dims(img, axis=2)
    img = np.append(edges, img, axis=2)
    return img


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from numpy import unique

    urls = get_urls()
    img = get_image(urls["url"][0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = transform(gray)
    plt.imshow(gray[:, :, 0], cmap="gray")
    # plt.imshow(gray[:, :, 1], cmap="gray")
    plt.show()
