from torch.utils.data import DataLoader
from data.functions import get_urls
from data.Dataset import PictureDataset
from data.mlFunctions import *
from sklearn.model_selection import train_test_split
from model import PictureColorizer
import torch
import torch.optim as optim
import pickle
import os

if __name__ == "__main__":

    # urls = get_urls()[:1_000]
    # urls = get_urls()[:40]

    # train_urls, val_urls = train_test_split(urls, test_size=0.2)
    # val_dataset = PictureDataset(val_urls)
    # train_dataset = PictureDataset(train_urls)

    # path = os.path.dirname(__file__) + '/custom_datasets/'
    # with open(path + 'val_dataset_40.pkl', "wb") as file:
    #     pickle.dump(val_dataset, file)
    # with open(path + 'train_dataset_40.pkl', "wb") as file:
    #     pickle.dump(train_dataset, file)
    

    path = os.path.dirname(__file__) + '/custom_datasets/'
    with open(path + 'val_dataset.pkl', 'rb') as file:
        val_dataset = pickle.load(file)
    with open(path + 'train_dataset.pkl', 'rb') as file:
        train_dataset = pickle.load(file)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PictureColorizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    epochs = 10
    ckpt_path = os.path.dirname(os.path.dirname(__file__)) + '/models/best.pt'
    print("Start Train")

    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, ckpt_path=ckpt_path, device=device, plotting=True)
