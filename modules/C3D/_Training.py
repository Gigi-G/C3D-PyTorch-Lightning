from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np

from ._C3D import C3D

class TrainC3D(L.LightningModule):
    """ TrainC3D class.
    
        Args:
            lr (float): Learning rate.
            model (C3D): C3D model.
            save_dir (str): Directory to save the model.
            save_epochs (int): Number of epochs to save the model.
    """
    
    def __init__(self, 
                lr:float,
                model:C3D,
                save_dir:str,
                save_epochs:int
                ) -> None:
        super(TrainC3D, self).__init__()
        
        self.lr = lr
        self.model = model
        self.save_dir = save_dir
        self.save_epochs = save_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self) -> Any:
        train_params = [
            {"params": self.model.get_1x_lr_params(), "lr": self.lr},
            {"params": self.model.get_10x_lr_params(), "lr": self.lr * 10},
        ]
        optimizer = SGD(train_params, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        probs = nn.Softmax(dim=1)(y_hat)
        preds = torch.max(probs, 1)[1]
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss)
        predictions = preds.cpu().numpy()
        labels = y.cpu().numpy()
        self.training_step_outputs.append({"predictions": predictions, "labels": labels})
        return {"loss": loss, "predictions": predictions, "labels": labels}
    
    def on_train_epoch_end(self) -> None:
        predictions = np.concatenate([o["predictions"] for o in self.training_step_outputs])
        labels = np.concatenate([o["labels"] for o in self.training_step_outputs])
        acc = np.sum(predictions == labels) / len(labels)
        self.log("train/acc", acc)
        self.training_step_outputs.clear()
        
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        probs = nn.Softmax(dim=1)(y_hat)
        preds = torch.max(probs, 1)[1]
        loss = self.criterion(y_hat, y).item()
        self.log("val/loss", loss)
        predictions = preds.cpu().numpy()
        labels = y.cpu().numpy()
        self.validation_step_outputs.append({"predictions": predictions, "labels": labels})
        return {"predictions": predictions, "labels": labels}
    
    def on_validation_epoch_end(self) -> None:
        predictions = np.concatenate([o["predictions"] for o in self.validation_step_outputs])
        labels = np.concatenate([o["labels"] for o in self.validation_step_outputs])
        acc = np.sum(predictions == labels) / len(labels)
        self.log("val/acc", acc)
        if self.current_epoch % self.save_epochs == 0:
            torch.save({
                "epoch": self.current_epoch + 1,
                "state_dict": self.model.state_dict(),
                "opt_dict": self.optimizers().state_dict(),
                "loss": self.criterion,
            }, f"{self.save_dir}/model_{self.current_epoch}.pth")
        self.validation_step_outputs.clear()
        
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        probs = nn.Softmax(dim=1)(y_hat)
        preds = torch.max(probs, 1)[1]
        predictions = preds.cpu().numpy()
        labels = y.cpu().numpy()
        self.test_step_outputs.append({"predictions": predictions, "labels": labels})
        return {"predictions": predictions, "labels": labels}
    
    def on_test_epoch_end(self) -> None:
        predictions = np.concatenate([o["predictions"] for o in self.test_step_outputs])
        labels = np.concatenate([o["labels"] for o in self.test_step_outputs])
        acc = np.sum(predictions == labels) / len(labels)
        self.log("test/acc", acc)
        self.test_step_outputs.clear()
        
        