from model.data.transforms import transform
import torch
from torchvision.transforms import Normalize

class PictureColorizer(torch.nn.Module):
    def __init__(self):
        super(PictureColorizer, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(4, 3, 3, padding=1)
        self.norm = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv6(x)
        x = torch.nn.functional.relu(x)
        x = self.norm(x)
        x = x - torch.min(x)
        x = x/torch.max(x)
        
        return x


def predict(model, gray):
    prepared = transform(gray)
    img = model(prepared)
    return img
