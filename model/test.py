from turtle import color
from data.functions import get_urls, get_image
from data.transforms import transform
import torch
from model import PictureColorizer
from matplotlib import pyplot as plt

model = PictureColorizer()
model.load_state_dict(torch.load("./models/best.pt", weights_only=True))
model.eval()

url = get_urls()[0]["url"]

image = get_image(url[0])
gray = torch.Tensor(transform(image))
gray = gray.transpose(0, 2)
gray = gray.transpose(1, 2)
colored = model(gray)

colored = colored.transpose(1, 2)
colored = colored.transpose(0, 2)
colored = colored.detach().numpy()

f, subplots = plt.subplots(1, 2)
subplots[0].imshow(image)
subplots[1].imshow(colored)
plt.show()
