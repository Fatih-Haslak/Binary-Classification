import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.GELU()

        # Eğer boyut değişimi varsa, residual bağlantıyı uygun hale getir
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)

        # Residual bağlantıyı ekleyin
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.resblock1 = ResidualBlock(16, 32)
        
        self.resblock2 = ResidualBlock(32, 64)
        
        self.resblock3 = ResidualBlock(64, 64)
        
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.relu4 = nn.GELU()
        
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        
        x = x.view(-1, 64 * 64 * 64)
        
        x = self.fc1(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x