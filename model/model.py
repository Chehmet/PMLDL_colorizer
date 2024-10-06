from data.transforms import transform
import torch


class PictureColorizer(torch.nn.Module):
    def __init__(self):
        super(PictureColorizer, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(512, 3, 3, padding=1)
        # self.fc1 = torch.nn.Linear(512, 256)
        # self.fc2 = torch.nn.Linear(256, 128)
        # self.fc3 = torch.nn.Linear(128, 64)
        # self.fc4 = torch.nn.Linear(64, 32)
        # self.fc5 = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = torch.nn.functional.relu(x)
        # x = self.fc2(x)
        # x = torch.nn.functional.relu(x)
        # x = self.fc3(x)
        # x = torch.nn.functional.relu(x)
        # x = self.fc4(x)
        # x = torch.nn.functional.relu(x)
        x = self.conv6(x)
        return x


def predict(model, gray):
    prepared = transform(gray)
    img = model(prepared)
    return img
