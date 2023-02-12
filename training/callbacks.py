import pytorch_lightning as pl
import torch
import torchvision

class OutputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, _ = batch
        data = x[:, 0, ...]
        z = pl_module.encoder(data)
        y = pl_module.lookup(z.unsqueeze(1)).squeeze(1)
        trainer.logger.experiment.add_histogram("encoder_histogram", y, global_step=trainer.global_step)
        trainer.logger.experiment.add_histogram("lookup_histogram", z, global_step=trainer.global_step)
        
        z2 = pl_module.encoder(torch.zeros_like(data).uniform_(-1, 1))
        y2 = pl_module.lookup(z2.unsqueeze(1)).squeeze(1)
        pl_module.log("lookup_diff_from_noise", abs(y2 - y).mean())
                   
class InputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            trainer.logger.experiment.add_histogram("image_histogram", x, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_histogram", y, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_normalized_histogram", pl_module.embed_normalizer(y), global_step=trainer.global_step)

class GenerateCallback(pl.Callback):
    def _get_train_images_part(self, data_loader):
        for img, _ in data_loader:
            return img[0, ...]
            
    def __init__(self, data_loader, every_n_epochs = 5):
        super().__init__()
        self.input_imgs = self._get_train_images_part(data_loader)
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                z = pl_module.encoder(input_imgs)
                reconst_imgs = pl_module.decoder(z)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
