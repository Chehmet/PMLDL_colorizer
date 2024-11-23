from data.transforms import lab_to_rgb
from data.functions import plot5pics, plot_loss, save_model
from tqdm import tqdm
import torch


def train_one_epoch(
    loader, 
    model_generator,
    optim_gen,
    loss_fn_gen,
    model_discriminator=None,
    optim_disc=None,
    loss_fn_disc=None,
    model_name='cnn',
    epoch_num=-1, 
    device='cpu', 
    L1_LAMBDA=100,
    ):

    loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_num}: train", leave=True)
    model_generator.train()
    if model_discriminator is not None:
        model_discriminator.train()

    generative_loss_sum = 0.0
    discriminative_loss_sum = 0.0

    # train loop
    for i, batch in loop:
        gray, colored = batch
        colored = colored.to(device)
        gray = gray.to(device)

        # forward pass
        outputs = model_generator(gray)

        if model_name=='gan':
            # discriminator: comparison with actual image and loss calculation
            D_real = model_discriminator(gray, colored)
            D_fake = model_discriminator(gray, outputs.detach())

            D_real_l = loss_fn_disc(D_real, torch.ones_like(D_real))
            D_fake_l = loss_fn_disc(D_fake, torch.zeros_like(D_fake))
            discriminative_loss = D_real_l + D_fake_l

            optim_disc.zero_grad()
            discriminative_loss.backward()
            optim_disc.step()
            discriminative_loss_sum += discriminative_loss.item()


        # generator: loss calculation
        generative_loss = loss_fn_gen(outputs, colored)
        if model_name == 'gan':
            D_fake = model_discriminator(gray, outputs)
            G_fake_l = loss_fn_gen(D_fake, torch.ones_like(D_fake))
            loss_scaled = generative_loss *L1_LAMBDA
            generative_loss = G_fake_l + loss_scaled

        optim_gen.zero_grad()
        generative_loss.backward()
        optim_gen.step()
        generative_loss_sum += generative_loss.item()

        loop.set_postfix({"loss": float(generative_loss)})
    
    if model_name == 'cnn':
        print('Train loss:', float(generative_loss_sum) / (i+1))
    else:
        print('Generator loss: ', generative_loss_sum/(i+1), '. Discriminator loss: ', discriminative_loss_sum/(i+1), sep='')

    return generative_loss_sum / (i+1), discriminative_loss_sum / (i+1)



def val_one_epoch(
    model_generator,
    loader,
    loss_fn_gen,
    model_name='cnn',
    model_discriminator=None,
    loss_fn_disc=None,
    plots_path="plots/",
    epoch_num=-1,
    device='cpu',
    L1_LAMBDA = 100,
):

    loop = tqdm(enumerate(loader, 1), total=len(loader), desc=f"Epoch {epoch_num}: val", leave=True)

    gen_loss_sum = 0.0
    disc_loss_sum = 0.0

    # keep images to plot them
    fake_imgs = []
    real_imgs = []
    gray_imgs = []

    with torch.no_grad():
        model_generator.eval()
        if model_discriminator is not None:
            model_discriminator.eval()
        
        
        # save 1 picture each batch
        for i, batch in loop:
            gray, colored = batch
            colored = colored.to(device)
            gray = gray.to(device)

            # forward pass
            outputs = model_generator(gray)

            if model_name == 'gan':
                D_real = model_discriminator(gray, colored)
                D_fake = model_discriminator(gray, outputs.detach())

                D_real_l = loss_fn_disc(D_real, torch.ones_like(D_real))
                D_fake_l = loss_fn_disc(D_fake, torch.zeros_like(D_fake))
                discriminative_loss = D_real_l + D_fake_l

                disc_loss_sum += discriminative_loss.item()

            # loss calculation
            generative_loss = loss_fn_gen(outputs, colored)
            if model_name == 'gan':
                D_fake = model_discriminator(gray, outputs)
                G_fake_l = loss_fn_gen(D_fake, torch.ones_like(D_fake))
                loss_scaled = generative_loss * L1_LAMBDA
                generative_loss = G_fake_l + loss_scaled
            gen_loss_sum += generative_loss.item()

            loop.set_postfix({"loss": float(gen_loss_sum)})

            # picture saving
            if model_name == 'cnn':
                gray_imgs.append(gray[0][1].squeeze().cpu())
                real_imgs.append(colored[0].permute(1, 2, 0).cpu())
                fake_imgs.append(outputs[0].permute(1, 2, 0).cpu())
            else:
                gray_imgs.append(gray[0].cpu()[0])
                real_imgs.append(lab_to_rgb(gray, colored)[0])
                fake_imgs.append(lab_to_rgb(gray, outputs.detach())[0])

    # save model result to compare it over epochs
    path = f'{plots_path}visual_progress_epoch_{epoch_num}.png'
    plot5pics(gray_imgs, real_imgs, fake_imgs, epoch_num, path)

    print('Val loss:',float(gen_loss_sum) / (i+1))
    return gen_loss_sum / (i+1), disc_loss_sum / (i+1)



def train(
    train_dataloader,
    val_dataloader,
    model_generative,
    optim_generative,
    loss_fn_generative,
    model_disc=None,
    optim_disc=None,
    loss_fn_disk=None,
    model_name='cnn',
    epochs=10,
    ckpt_path="models/best.pt",
    plots_path="plots/",
    device='cpu',
    save_freq=2,
):
    # check for any errors
    if model_name not in ['gan', 'cnn']:
        raise Exception("Error. Unsupportabble type of model (model_name).")
    
    # keep previous best loss to receive the 'best.pth' model with the least val loss
    prev_best = float("inf")

    # save losses for plotting
    losses_train_gen = []
    losses_train_disc = []
    losses_val_gen = []     
    losses_val_disc = []

    for epoch in range(epochs):
        # train
        generative_loss, disc_loss = train_one_epoch(
            train_dataloader,
            model_generative,
            optim_generative,
            loss_fn_generative,
            model_disc,
            optim_disc,
            loss_fn_disk,
            model_name=model_name,
            epoch_num=epoch,
            device=device,
            )
        
        if model_name == 'gan':
            losses_train_disc.append(disc_loss)
        losses_train_gen.append(generative_loss)

        # validate
        generative_loss, disc_loss = val_one_epoch(
            model_generative,
            val_dataloader,
            loss_fn_generative,
            model_name=model_name,
            model_discriminator=model_disc,
            loss_fn_disc=loss_fn_disk,
            plots_path=plots_path,
            epoch_num=epoch,
            device=device,
            )
        if model_name == 'gan':
            losses_val_disc.append(disc_loss)
        losses_val_gen.append(generative_loss)

        # model and model's output saving because intermediate results of Colorization
        # might be better than final
        if epoch % save_freq == 0:
            filename = ckpt_path+f'model{epoch}.pth'
            if model_name == 'cnn':
                save_model(model_generative, optim_generative, epoch, filename)
            else:
                save_model(model_generative, optim_generative, epoch, filename, model_disc, optim_disc)


        if generative_loss < prev_best:
            prev_best = generative_loss
            filename = ckpt_path + 'best.pth'

            if model_name == 'cnn':
                save_model(model_generative, optim_generative, epoch, filename)
            else:
                save_model(model_generative, optim_generative, epoch, filename, model_disc, optim_disc)

    # plot losses over epochs
    filename = plots_path + 'train_val_loss.png'
    plot_loss(losses_train_gen, losses_val_gen, losses_train_disc, losses_val_disc, filename)
