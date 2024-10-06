import cv2
import numpy as np
from data.transforms import transform
from data.functions import get_image
from tqdm import tqdm
import torch


def train_one_epoch(model, loader, optimizer, loss_fn, epoch_num=-1):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()

    for i, batch in loop:
        colored, gray = batch
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(gray)

        # loss calculation
        loss = loss_fn(outputs, colored)
        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        loop.set_postfix({"loss": float(loss)})


def val_one_epoch(
    model,
    loader,
    loss_fn,
    best_so_far=0.0,
    best=float("inf"),
    ckpt_path="./models/best.pt",
):

    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )

    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            colored, gray = batch

            # forward pass
            outputs = model(gray)

            # loss calculation
            loss = loss_fn(outputs, colored)

            loop.set_postfix({"mse": float(loss)})

        if loss < best:
            torch.save(model.state_dict(), ckpt_path)
            return loss

    return best_so_far


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    epochs=10,
    ckpt_path="./models/best.pt",
):
    best = float("inf")
    prev_best = best
    counter = 0
    for epoch in range(epochs):
        train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
        best = val_one_epoch(
            model, val_dataloader, loss_fn, epoch, best_so_far=best, ckpt_path=ckpt_path
        )
        if prev_best - best <= 0.0000001:

            counter += 1
        else:
            counter = 0
        if best < prev_best:
            prev_best = best
        if counter >= 5:
            break


def collate_batch(batch):
    colored_list, gray_list = [], []
    for url in batch:
        img = get_image(url)
        gray = transform(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        colored_list.append(img)
        gray_list.append(gray)
    colored_list = torch.tensor(colored_list)
    gray_list = torch.tensor(gray_list)

    return np.array(colored_list), np.array(gray_list)
