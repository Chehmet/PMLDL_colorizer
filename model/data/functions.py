from functools import lru_cache
# from nis import cat
import urllib.request
import cv2
import numpy as np
import polars as pl
import requests


@lru_cache
def check_url(url: str) -> bool:
    response = requests.head(url)
    # print(f"checking {url}")

    return response.status_code == 200


def get_image(url):

    req = requests.get(url).content
    # print(req)
    arr = np.asarray(bytearray(req), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_urls() -> pl.DataFrame:
    urls = pl.read_parquet(
        "hf://datasets/Chr0my/public_flickr_photos_license_1/**/*.parquet",
        columns=["url"],
    )

    # urls = urls.select(pl.col("url"))
    return urls


def adjust_brightness(image: np.ndarray, brightness_factor: float) -> np.ndarray:
    """
    Adjust the brightness of an image.

    Args:
        image (np.ndarray): Input image as a NumPy array (H, W, C) with values in range [0, 255].
        brightness_factor (float): Factor to adjust brightness (e.g., 1.0 keeps original, >1.0 brightens, <1.0 darkens).

    Returns:
        np.ndarray: Brightness-adjusted image as a NumPy array.
    """
    # Clip values to ensure they remain in the valid range [0, 255] after adjustment
    adjusted_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    return adjusted_image


def increase_color(image: np.ndarray, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image
