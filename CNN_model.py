import torch
from torch import nn
import math

## Simple CNN to be implemented in the classifier

class CNNetwork(nn.Module):

    def __init__(self, fine_tune='None', height=12):
        super().__init__()
        #2 conv blocks / dropout/ flatten / linear / dropout/ softmax
        self.fine_tune = fine_tune
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05)
        )

        h = math.ceil((height/8)+(5/4))

        self.flatten = nn.Flatten() # flatten from CNN to Linear

        # (channels, height, length)
        # 3 -> 12/13 MFCCs, 6 -> 40 MFCC , 9 -> 60MFCC/64 Melspec
        x = 6

        self.linear1 = nn.Linear(64*x*3, 256)  # for MelSpec

        self.linear2 = nn.Linear(256,6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        if self.fine_tune == 'conv':
            with torch.no_grad():
                x = self.conv1(input_data)
                x = self.conv2(x)
        elif self.fine_tune == 'linear':
            with torch.no_grad():
                x = self.conv1(input_data)
                x = self.conv2(x)
                x = self.conv3(x)
        else:
            x = self.conv1(input_data)
            x = self.conv2(x)
            x = self.conv3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        predictions = logits

        return predictions

if __name__ == "__main__":
    cnn = CNNetwork()
    #1 = channels, 64 = freq axis (number of mel bands), 44 = time axis
    # if no cuda --> dont put in
    loss = nn.CrossEntropyLoss
    summary(cnn, (1, 12, 16))
