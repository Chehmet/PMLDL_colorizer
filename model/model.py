from data.transforms import rgb_to_gray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.norm = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        
        # Upsampling layers to increase the spatial dimensions
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(64)
        
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(32, 3, kernel_size=3, padding='same')
        self.batchnorm6 = nn.BatchNorm2d(2)


    def forward(self, x):
        # Initial convolution layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.batchnorm2(x)
        
        x = F.relu(self.conv5(x))
        x = self.batchnorm3(x)
        
        # Upsampling and convolution layers with ReLU activations
        x = self.upsample1(x)
        x = F.relu(self.conv6(x))
        x = self.batchnorm4(x)
        
        x = self.upsample2(x)
        x = F.relu(self.conv7(x))
        x = self.batchnorm5(x)
        
        x = F.relu(self.conv8(x))
        
        x = F.relu(self.conv9(x))
        x = self.norm(x)
        x = x - torch.min(x)
        x = x/torch.max(x)
        return x

def predict(model, gray):
    prepared = rgb_to_gray(gray)
    img = model(prepared)
    return img


def encoder(in_channels=0, out_channels=0, first=False):
    if first:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

def decoder(in_channels=0, out_channels=0, dropout=False):
    if dropout:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(),
        )
    else:
        return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels, kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(out_channels),
    )


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder1 = encoder(in_channels=1, out_channels=64, first = True)
        self.encoder2 = encoder(in_channels=64, out_channels=128)
        self.encoder3 = encoder(in_channels=128, out_channels=256)
        self.encoder4 = encoder(in_channels=256, out_channels=512)
        self.encoder5 = encoder(in_channels=512, out_channels=512)
        self.encoder6 = encoder(in_channels=512, out_channels=512)
        self.encoder7 = encoder(in_channels=512, out_channels=512)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=4,stride=2,padding=1),
            nn.ReLU()
        )
        self.decoder1 = decoder(in_channels=512, out_channels=512, dropout=True)
        self.decoder2 = decoder(in_channels=1024, out_channels=512, dropout=True)
        self.decoder3 = decoder(in_channels=1024, out_channels=512, dropout=True)
        self.decoder4 = decoder(in_channels=1024, out_channels=512)
        self.decoder5 = decoder(in_channels=1024, out_channels=256)
        self.decoder6 = decoder(in_channels=512, out_channels=128)
        self.decoder7 = decoder(in_channels=256, out_channels=64)

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 2,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )

    def encode(self,x):
        skips = []
        x = self.encoder1(x)
        skips.append(x)
        x = self.encoder2(x)
        skips.append(x)
        x = self.encoder3(x)
        skips.append(x)
        x = self.encoder4(x)
        skips.append(x)
        x = self.encoder5(x)
        skips.append(x)
        x = self.encoder6(x)
        skips.append(x)
        x = self.encoder7(x)
        skips.append(x)
        x = self.conv1(x)
        return x, skips

    def decode(self, x, skips):
        x = self.decoder1(x)
        x = F.relu(x)
        x = torch.cat([x,skips[6]],1)
        x = self.decoder2(x)
        x = torch.cat([x,skips[5]],1)
        x = F.relu(x)
        x = self.decoder3(x)
        x = torch.cat([x,skips[4]],1)
        x = F.relu(x)
        x = self.decoder4(x)
        x = torch.cat([x,skips[3]],1)
        x = F.relu(x)
        x = self.decoder5(x)
        x = torch.cat([x,skips[2]],1)
        x = F.relu(x)
        x = self.decoder6(x)
        x = torch.cat([x,skips[1]],1)
        x = F.relu(x)
        x = self.decoder7(x)
        x = torch.cat([x,skips[0]],1)
        x = F.relu(x)
        return x

    def forward(self,x):
        x, skips = self.encode(x)
        out = self.decode(x,skips)
        return self.conv2(out)


class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)
    
    
class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(Discriminator,self).__init__()
        dims = [64, 128, 256, 512]
        self.first = nn.Sequential(
            nn.Conv2d(3,dims[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = dims[0]
        for dim in dims[1:]:
            layers.append(
                CNNBlock(in_channels, dim, stride=1 if dim == dims[len(dims)-int(1)] else 2),
            )
            in_channels = dim
        layers.append(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = self.first(x)
        return self.model(x)