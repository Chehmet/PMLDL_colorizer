from calendar import c
from re import sub
from model.data.functions import get_urls, get_image
from model.data.transforms import rgb_to_gray, lab_to_rgb, rgb_to_lab, increase_color
import torch
from torchvision import transforms
from model.model import CNN, Generator, Discriminator
from matplotlib import pyplot as plt
import cv2
import numpy as np

def colorize_image_gan(image):
    L = torch.Tensor(rgb_to_lab(image).astype("float32")).to(device)
    L = L.permute(2, 0, 1)
    L = L[[0],...] / 50. - 1.
    L = L.unsqueeze(0)

    with torch.no_grad():
        colored = model_gan(L)
        
    colored = lab_to_rgb(L, colored.detach())[0]
    if np.min(colored) < 0:
        colored -= np.min(colored)
        colored /= np.max(colored)
    return colored

resize_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((2048, 2048)),
        transforms.ToTensor(),
    ]
)

model = CNN()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(
    torch.load(
        "./models/cnn/best_model_after42.pt",
        map_location=torch.device(device),
        weights_only=True,
    )
)
model.eval()

model_gan = Generator()
model_gan.load_state_dict(
    torch.load("./models/gan/best.pth", weights_only=True, map_location=torch.device("cpu"))[
        "model_generator"
    ]
)

model_gan.eval()

url = get_urls()["url"][1000]

image = get_image(url)

gray_with_edges = torch.Tensor(rgb_to_gray(image))
gray_with_edges = gray_with_edges.transpose(0, 2)
gray_with_edges = gray_with_edges.transpose(1, 2)
gray_with_edges = resize_transform(gray_with_edges)
gray_with_edges = gray_with_edges.unsqueeze(0)

image = resize_transform(image)

image = image.transpose(1, 2)
image = image.transpose(0, 2)
image = image.detach().numpy()

colored_CNN = model(gray_with_edges)
colored_CNN = colored_CNN.squeeze(0)
colored_CNN = colored_CNN.transpose(1, 2)
colored_CNN = colored_CNN.transpose(0, 2)
colored_CNN = colored_CNN.detach().numpy()
colored_CNN = 1 - colored_CNN




colored_GAN_from_CNN = colorize_image_gan(colored_CNN)


colored_GAN_from_original = colorize_image_gan(image)



f, subplots = plt.subplots(2, 3)
subplots[0][0].imshow(image)
subplots[0][0].set_title("Original")
subplots[0][1].imshow(colored_CNN)
subplots[0][1].set_title("Colorized with CNN")
subplots[0][2].imshow(colored_GAN_from_CNN)
subplots[0][2].set_title("Colorized with GAN after CNN")
subplots[1][0].imshow(colored_GAN_from_original)
subplots[1][0].set_title("Colorized with GAN from original")
subplots[1][1].imshow(increase_color(colored_GAN_from_original, 1.2))
subplots[1][1].set_title("Colorized with GAN from original with color enhancement")
subplots[1][2].imshow(increase_color(colored_GAN_from_CNN, 1.2))
subplots[1][2].set_title("Colorized with GAN after CNN with color enhancement")
plt.show()
