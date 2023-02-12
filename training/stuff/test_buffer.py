import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from leap_sd import LEAPBuffer

size_in = 3
size_out = 509248

def get_datamodule(batch_size: int):
    class FakeDataset(Dataset):
        def __init__(self, amount):
            self.amount = amount
            self.x = torch.zeros(amount, size_in).uniform_(0, 1)
            self.y = torch.zeros(amount, size_out).uniform_(0, 1)

        def __getitem__(self, index):
            return self.x[index, ...], self.y[index, ...]
        
        def __len__(self):
            return self.amount

    class ImageWeights(pl.LightningDataModule):
        def __init__(self, batch_size: int):
            super().__init__()
            self.num_workers = 16
            self.batch_size = batch_size
            
        def prepare_data(self):
            pass

        def setup(self, stage):
            pass
            
        def train_dataloader(self):
            dataset = FakeDataset(25)
            return DataLoader(dataset, num_workers = self.num_workers, batch_size = self.batch_size, drop_last = True)

        def teardown(self, stage):
            # clean up after fit or test
            # called on every process in DDP
            pass
    
    dm = ImageWeights(batch_size = batch_size)
    
    return dm

class TestModel(pl.LightningModule):
    def __init__(self, size_in: int, size_out: int, hidden_size: int, learning_rate: float):
        super().__init__()
        self.buf = LEAPBuffer(size_in, size_out, hidden_size, 1, 0.01)
        self.criterion = torch.nn.L1Loss()
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        result = self.buf(x)
        print("result x", result[0])
        print("result y", y[0])
        print("diff", abs(y[0, ...] - y[1, ...]).mean())
        loss = self.criterion(result, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5),
            "monitor": "train_loss",
            "interval": "epoch"
        }
        return [optimizer], [scheduler]

def main():
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('medium')

    hidden_size = 1024
    model = TestModel(size_in, size_out, hidden_size, 1e-4)
    dm = get_datamodule(10)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    from pytorch_lightning.loggers import WandbLogger
    trainer = pl.Trainer(auto_lr_find=True, devices=1, accelerator="gpu", callbacks = [lr_monitor], log_every_n_steps=2, max_epochs=100)
    # trainer = pl.Trainer(devices=1, accelerator="gpu", logger = WandbLogger(project="LEAP_Lora_BufferTest"), callbacks = [lr_monitor], log_every_n_steps=2, max_epochs=1000)
    # trainer.tune(model, dm)
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()