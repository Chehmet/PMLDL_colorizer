from torch.utils.data import DataLoader
from data.functions import get_urls
from data.Dataset import PictureDataset
from data.mlFunctions import *
from sklearn.model_selection import train_test_split
from model import PictureColorizer
import torch
import torch.optim as optim

if __name__ == "__main__":
    urls = get_urls()[:1_000]

    train_urls, val_urls = train_test_split(urls, test_size=0.2)
    val_dataloader = PictureDataset(val_urls)
    train_dataloader = PictureDataset(train_urls)
    model = PictureColorizer()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    epochs = 10
    print("start train")
    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs)
