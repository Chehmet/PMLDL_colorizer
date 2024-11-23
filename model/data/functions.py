from functools import lru_cache
import cv2
import numpy as np
import polars as pl
import requests
import matplotlib.pyplot as plt
import json
import torch


@lru_cache
def check_url(url: str) -> bool:
    """ Check if an image is available."""
    response = requests.head(url)
    return response.status_code == 200


def get_image(url: str) -> np.ndarray:
    """ Get RGB image from url in np.ndarray type."""
    req = requests.get(url).content
    arr = np.asarray(bytearray(req), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_urls() -> pl.DataFrame:
    """ Receive a list of images' urls in the format similar to csv. """
    urls = pl.read_parquet(
        "hf://datasets/Chr0my/public_flickr_photos_license_1/**/*.parquet",
        columns=["url"],
    )
    return urls



def plot5pics(gray, color, output, epoch: int, path: str) -> None:
    """ Plot a grid of images.
        1st raw: 5 samples of gray images.
        2nd raw: 5 corresponding samples of model outputs.
        3rd raw: 5 corresponding samples of correctly colored images."""
    
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        # ax.imshow(gray[i][0].cpu(), cmap='gray')
        ax.imshow(gray[i].cpu(), cmap='gray')

        ax.axis("off")
        ax.set_title('Gray images')
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(output[i])
        ax.axis("off")
        ax.set_title(f'Epoch {epoch}')
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(color[i])
        ax.axis("off")
        ax.set_title('Ground truth')
    plt.plot()
    plt.savefig(path)

def plot_loss(losses_train_gen: list, losses_val_gen: list, 
              losses_train_disc: list, losses_val_disc: list, 
              path: str) -> None:
    """ Plot losses over epochs.
        Independently of model type train loss and validation loss will be plotted.
        Discriminator loss will be plotted if a passed list (losses_train_disc) is not empty
        (if it was updated during training). """
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_train_gen, label='Training generative loss')
    plt.plot(losses_val_gen, label='Validation generative loss')
    if len(losses_train_disc) != 0:
        plt.plot(losses_train_disc, label='Training discriminative loss')
        plt.plot(losses_val_disc, label='Validation discriminative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.savefig(path)
  


def get_config(path: str) -> dict:
    """ Receive a dictionary with setted parameters from the main config - vars_config.json."""
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def save_model(model_gen, opt_gen, epoch, path: str, model_disc=None, opt_disc=None) -> None:
    """ Save model's and optimizer's state.
        Keep epoch number also for tracking and further training."""
    
    if model_disc is not None:
        torch.save({'model_generator': model_gen.state_dict(), 
                            'model_discriminator': model_disc.state_dict(),
                            'optim_generative':opt_gen.state_dict(),
                            'optim_discriminative': opt_disc.state_dict(), 
                            'epoch': epoch}, 
                            path)
    else:
        torch.save({'model_generator': model_gen.state_dict(), 
                    'optim_generative':opt_gen.state_dict(),
                    'epoch': epoch}, 
                    path)

