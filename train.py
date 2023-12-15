import os
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data.dataloader import DogsAndCatsDataLoader
from model.cnn_model import CNNModel

import warnings

# Belirli bir uyarı tipini ve içeriğini filtreleme
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*", category=UserWarning)

# torchvision modülünü içe aktarın
import torchvision

class Classification(L.LightningModule):
    def __init__(
            self,
            lr: float = 0.0001,
            b1: float = 0.9,
            b2: float = 0.999,
        ):
            super().__init__()
            self.save_hyperparameters()
            self.automatic_optimization = False
            self.cnn = CNNModel()
            criterion = nn.BCELoss()
            self.criterion=criterion #loss fonc

    def forward(self, z):    
        return self.cnn(z) #torch.image generator forward
    
    def training_step(self, batch,batch_idx):
        #image, #label
        img, label = batch
        label = label.view(-1, 1).float()
        optimizer_model= self.optimizers()
        self.toggle_optimizer(optimizer_model)
        self.predicted_label = self(img) #img
        #calculate loss
        loss = self.criterion(self.predicted_label, label)
        self.log("Loss", loss, prog_bar=True) #loggla
        self.manual_backward(loss,retain_graph=True)# backward
        optimizer_model.step()
        optimizer_model.zero_grad()
        self.untoggle_optimizer(optimizer_model)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1).float()
        #x=self.validation_z.type_as(x)
        predict = self.cnn(x)
        
        loss = self.criterion(predict,y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_model = torch.optim.Adam(self.cnn.parameters(), lr=lr, betas=(b1, b2))
    
        return [optimizer_model], []
    
if __name__ == "__main__":
    root_folder = r"C:\Users\90546\Desktop\dataset"
    datacatsdogs = DogsAndCatsDataLoader(root_folder)
    model=Classification()    
    train,val=datacatsdogs.load_data()
   
    trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=40,
    #strategy="ddp_find_unused_parameters_true",
    #num_nodes=1
    )
    trainer.fit(model, train, val)
    trainer.save_checkpoint("best_model.ckpt")
 