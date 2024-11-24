from torch.utils.data import Dataset
from torch import max
from torchvision import transforms
import cv2
import os
import sys
import numpy as np
from PIL import Image

# set project root to sys paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.path_config import get_root_path
from model.data.transforms import rgb_to_gray, rgb_to_lab


class PictureDataset(Dataset):
    def __init__(self, path=None, image_size=1024, mode='train', convert_to='gray', amount=2000):

        # set the dataset's length
        amount = int(amount)
        self.n = amount

        # assign paths to each image from pictures/train or pictures/val folder
        if not path:
            path = get_root_path() + '/pictures'
            self.paths = [os.path.join(path, mode, img) for img in os.listdir(os.path.join(path, mode))]
            self.paths = self.paths[:amount]

        # add transform: PIL, Resize, and ToTensor
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
        ])
        self.resize = transforms.Resize((image_size, image_size))

        # convertation depends on model type (to Gray if CNN, to LAB if GAN)
        self.convert_to = convert_to

    def __len__(self):
        return int(self.n)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.convert_to == 'gray':
            # convert to gray, apply transformation, normalize
            # return gray image and original image 
            gray = rgb_to_gray(image)
            image, gray = self.resize_transform(image), self.resize_transform(gray)
            image = image/max(image)
            gray = gray/max(gray)
            return gray, image

        elif self.convert_to == 'lab':
            # resize, convert to lab, to tensor, split to L and ab
            # return L (brightness channel) and ab (chroma information)
            image = Image.fromarray(image) 
            image = self.resize(image)
            image = np.array(image)
            lab = rgb_to_lab(image).astype("float32")
            lab = transforms.ToTensor()(lab)
            L = lab[[0], ...] / 50. - 1.
            ab = lab[[1, 2], ...] / 110.
            return L, ab
        
        else:
            # other types are not supportable
            raise Exception("Error occured. Cannot recognize convert type.")
        