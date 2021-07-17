from torch import nn
from torchsummary import summary

## Simple CNN to be implemented in the classifier

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        #2 conv blocks / dropout/ flatten / linear / dropout/ softmax
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
            nn.MaxPool2d(kernel_size=2, stride=2)
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
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        #10 here is no of classes of audio (change to fit how many languages)
        #fix
        #self.linear = nn.Linear(128 * 5 * 2, 2)
        self.linear1 = nn.Linear(32 * 17 * 5, 256)
        self.linear2 = nn.Linear(256,6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.linear1(x)
        logits = self.linear2(x)
        predictions = logits
        return predictions

if __name__ == "__main__":
    cnn = CNNNetwork()
    #1 = channels, 64 = freq axis (number of mel bands), 44 = time axis
    # if no cuda --> dont put in
    loss = nn.CrossEntropyLoss
    summary(cnn, (1, 64, 16))
