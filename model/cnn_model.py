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
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.resblock1 = ResidualBlock(32, 64)
        
        self.resblock2 = ResidualBlock(64, 128)
        
        self.resblock3 = ResidualBlock(128, 128)
        
        self.fc1 = nn.Linear(131072, 256)
        self.batchnorm4 = nn.LayerNorm(256)
        self.batchnorm5 = nn.LayerNorm(128)
        self.batchnorm6 = nn.LayerNorm(64)
        self.relu4 = nn.GELU()
        self.fc1_1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu1(x)
      
        x = self.pool1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        # print(x.size())
        x = x.view(x.size(0),-1) #önemli nokta.
        # print(x.size())
        x = self.fc1(x)
        #=nn.Linear(131072, 256)
        x = self.batchnorm4(x)
      
        x = self.relu4(x)
        
        x = self.fc1_1(x)
        x = self.batchnorm5(x)
        x = self.relu4(x)
        
        x = self.fc2(x)
        x = self.batchnorm6(x)
        x = self.relu4(x)
        
        x = self.fc3(x)
        x = self.relu4(x)
        
        x = self.fc4(x)
        x = self.relu4(x)
        
        x=self.fc5(x)
      
   
        x = self.sigmoid(x)

      
        return x
