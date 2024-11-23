import cv2
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """ Convert image representation from RGB to grayscale: extract edge lines via Canny. """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)
    edges = np.expand_dims(edges, axis=2)
    img = np.expand_dims(img, axis=2)
    img = np.append(edges, img, axis=2)
    return img


def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """ Convert image RGB representation to LAB. """
    return rgb2lab(image)


def increase_color(image: np.ndarray, saturation: float) -> np.ndarray:
    """ Increase colorfulness of image. 
        Since CNN poorly coloizes images but can assign correct colors
        colorfulness addition makes it better. """
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if np.min(enhanced_image) < 0:
        enhanced_image -= np.min(enhanced_image)
        enhanced_image /= np.max(enhanced_image)
    return enhanced_image