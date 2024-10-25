from functools import lru_cache
from nis import cat
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
