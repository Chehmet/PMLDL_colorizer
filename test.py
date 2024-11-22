from calendar import c
from model.data.functions import get_urls, get_image
from model.data.transforms import rgb_to_gray, lab_to_rgb, rgb_to_lab
import torch
from torchvision import transforms
from model.model import CNN, Generator, Discriminator
from matplotlib import pyplot as plt
import cv2

resize_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
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

url = get_urls()["url"][0]

image = get_image(url)
print("got image")
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
print("got colored CNN")

L = torch.Tensor(rgb_to_lab(image).astype("float32")).to('cpu')
L = L.permute(2, 0, 1)
L = L[[0],...] / 50. - 1.
L = L.unsqueeze(0)

with torch.no_grad():
    colored_GAN = model_gan(L)

print(colored_GAN.shape)
colored_GAN = lab_to_rgb(L, colored_GAN.detach())[0]


print("got colored GAN")



f, subplots = plt.subplots(1, 3)
subplots[0].imshow(image)
subplots[1].imshow(colored_CNN)
subplots[2].imshow(colored_GAN)
plt.show()
