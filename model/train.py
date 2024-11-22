from torch.utils.data import DataLoader
from data.functions import get_config
from data.Dataset import PictureDataset
from data.mlFunctions import *
from model import CNN, Generator, Discriminator
import torch
import torch.optim as optim
import os
import sys
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.path_config import get_model_path, get_plots_path, get_main_config_path

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    config = get_config(get_main_config_path())
    
    epochs = config['epochs']
    batch_size = config['batch_size']
    pic_size = config['pic_size']
    train_val_split = config['train_val_split']
    model_name = config['model']
    pictures_num = config['pictures_num']

    ckpt_path = get_model_path(model_name) + '/'
    plots_path = get_plots_path(model_name) + '/'

    if model_name == 'cnn':
        convert_to = 'gray'
    elif model_name == 'gan':
        convert_to = 'lab'
    else:
        raise Exception("Error. Unsupportable type of model (model_name).")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    train_dataset = PictureDataset(mode='train', convert_to=convert_to, amount=pictures_num*(1-train_val_split), image_size=pic_size)
    val_dataset = PictureDataset(mode='val', convert_to=convert_to, amount=pictures_num*train_val_split, image_size=pic_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model_name == 'cnn':
        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        train(train_dataloader, val_dataloader, 
              model, optimizer, loss_fn, 
              model_name=model_name, epochs=epochs, ckpt_path=ckpt_path, plots_path=plots_path, device=device, save_freq=2)

    else:
        model_discriminator = Discriminator(in_channels=3).to(device)
        model_generator = Generator().to(device)
        optim_disc = optim.Adam(model_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optim_gen = optim.Adam(model_generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        loss_fn_gen = torch.nn.L1Loss()
        loss_fn_disc = torch.nn.BCEWithLogitsLoss()

        train(train_dataloader, val_dataloader, 
              model_generator, optim_gen, loss_fn_gen, 
              model_discriminator, optim_disc, loss_fn_disc,
              model_name, epochs, ckpt_path, plots_path, device, 1)
