from tabnanny import check
from data.functions import get_urls, check_url
from torch.utils.data import DataLoader
from data.mlFunctions import *
from sklearn.model_selection import train_test_split
from model import PictureColorizer
import torch
import torch.optim as optim
import polars as pl

if __name__ == "__main__":
    urls = get_urls()[:100]
    urls = urls.filter(pl.col("url").map_elements(check_url, return_dtype=bool))
    train_urls, val_urls = train_test_split(urls, test_size=0.2)
    val_dataloader = DataLoader(
        val_urls, batch_size=32, shuffle=True, collate_fn=collate_batch
    )
    train_dataloader = DataLoader(
        train_urls, batch_size=32, shuffle=True, collate_fn=collate_batch
    )
    model = PictureColorizer()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    epochs = 10
    print("start train")
    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs)
