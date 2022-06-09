from typing import Any

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from xlwic_model import XLWiCModel
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class BasePLModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(conf)
        self.model = XLWiCModel(conf)
        self.test_accuracy = None
        self.tokenizer = AutoTokenizer.from_pretrained(conf.model.pretrained_model_name)
        

    def forward(self, input_ids, indices_mask, labels=None) -> dict:
        return self.model(input_ids, indices_mask, labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output["loss"])
        sch:LinearWarmupCosineAnnealingLR = self.lr_schedulers()
        sch.step()
        lr = sch.get_last_lr()[0]
        self.log('lr', lr, sync_dist=True, prog_bar=True)
        
        return forward_output["loss"]
    
    def accuracy(self, predictions, labels):
        return accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy())
        
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**batch)
        self.log("val_loss", forward_output["loss"])
        predictions = forward_output['predictions']
        labels = batch['labels']
        return predictions, labels

    def validation_epoch_end(self, outputs) -> None:
        predictions, labels = zip(*outputs)
        predictions = torch.cat(predictions, 0).long().squeeze()
        labels = torch.cat(labels, 0).long().squeeze()
        
        acc = self.accuracy(predictions, labels)
        self.log('val_acc', acc, on_epoch=True)
        print()
        print(f'val_acc: {acc:.2f}')
        print()

    def on_test_epoch_start(self) -> None:
        self.test_accuracy = None

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        forward_output = self.forward(**batch)
        self.log("test_loss", forward_output["loss"])
        predictions = forward_output['predictions']
        labels = batch['labels']
        return predictions, labels
    
    def test_epoch_end(self, outputs) -> None:
        self.test_accuracy = None
        predictions, labels = zip(*outputs)
        predictions = torch.cat(predictions, 0).long().squeeze()
        labels = torch.cat(labels, 0).long().squeeze()
        
        acc = self.accuracy(predictions, labels)
        self.log('test_acc', acc)
        self.test_accuracy = acc

    def configure_optimizers(self):
        # params = map(lambda elem: elem[1], filter(lambda item: not item[0].startswith('encoder'), self.model.named_parameters()))
        params = self.model.parameters()
        optimizer = AdamW(params, lr=self.hparams.train.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, 1500, 1500 * self.hparams.train.pl_trainer.max_epochs)
        return {
            'optimizer':optimizer, 
            'lr_scheduler': 
            {
                'scheduler':scheduler
            }
        }