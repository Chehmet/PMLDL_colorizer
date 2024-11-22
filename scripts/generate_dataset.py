import os
from torchvision import transforms
import sys
import json

# set project root to sys paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.path_config import get_root_path, get_main_config_path
from model.data.functions import get_urls, check_url, get_image

def main():

    config_path = get_main_config_path()
    with open(config_path, 'r') as file:
        config = json.load(file)

    amount = config['pictures_num']
    val_split = config['train_val_split']
    pic_size = config['pic_size']

    train_amount = amount * (1 - val_split)
    val_amount = amount * val_split

    train_path = get_root_path() + '/pictures/train/'
    val_path = get_root_path() + 'pictures/val/'
    
    urls = get_urls()["url"]
    counter = 0

    resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((pic_size, pic_size)),
        ])

    for i in range(len(urls)):
        url = urls[i]
        if counter == train_amount:
            break
        if check_url(url):
            img = get_image(url)
            if img is not None:
                img = resize_transform(img)
                path = train_path + 'image' + str(counter) + '.png'
                img.save(path)
                counter += 1
    counter = 0
    for j in range(i, len(urls)):
        url = urls[j]
        if counter == val_amount:
            break
        if check_url(url):
            img = get_image(url)
            if img is not None:
                img = resize_transform(img)
                path = val_path + 'image' + str(counter) + '.png'
                img.save(path)
                counter += 1


if __name__ == "__main__":
    main()