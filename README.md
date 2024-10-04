# Black & White Photo Colorization and Restoration

## Team Members
- **Chulpan Valiullina**
- **Anastasia Pichugina**
- **Saveliy Khlebnov**

## Project Overview
This project aims to develop a custom model for colorizing black and white photos.
## Dataset
We are using the [Public Flickr Photos License 1 Dataset](https://huggingface.co/datasets/Chr0my/public_flickr_photos_license_1) from Hugging Face, which contains **120 million colorful images**. These images will be processed to create grayscale versions for training purposes.

## Steps
1. **Data Preprocessing**: Convert the colorful images to grayscale to simulate black and white photos.
2. **Model Training**: Train a custom model to colorize grayscale images using the processed dataset.
3. **Model Evaluation**: Test Apply the trained model to colorize black and white photos.
4. **Enhancement** (Optional): If time allows, we will expand the model to include photo restoration features (e.g., handling faded or damaged photos).

## Goals
- Build a robust colorization model.
- Optionally, integrate photo restoration features if time permits.

## Work distribution
1. Saveliy and Chulpan - model building and evaluation
2. Anastasia - API
## License
This project uses Hugging Face's [licence](https://spdx.org/licenses/CC-BY-NC-SA-3.0).
