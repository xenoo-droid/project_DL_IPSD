import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    # dibaca train.py/test.py buat getData; ikut M9: input 100x100, BGR (cv.imread), piksel 0-1
    PREPROCESS = {"size": 100, "bgr": True, "normalize": False}

    def __init__(self, input_dim=200, input_c=3, num_classes=6, hidden_dim=128, dropout=0.5, device='cpu'):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.fc = nn.Linear(in_features=50 * 9 * 9, out_features=num_classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.fc(torch.flatten(x, 1))
        x = self.Softmax(x)
        return x