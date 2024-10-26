import time
from torch.utils.data import DataLoader
from data.functions import get_urls
from data.Dataset import PictureDataset
from data.mlFunctions import *
from sklearn.model_selection import train_test_split
from model import PictureColorizer
import torch
import torch.optim as optim

def train(model, train_loader, val_loader, optimizer, loss_fn, epochs):
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        end_time = time.time()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    urls = get_urls()[:1000]
    train_urls, val_urls = train_test_split(urls, test_size=0.2)
    val_dataloader = PictureDataset(val_urls)
    train_dataloader = PictureDataset(train_urls)
    model = PictureColorizer()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    epochs = 10
    print("Start training")
    
    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs)