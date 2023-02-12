import pytorch_lightning as pl
import torch

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
        z = pl_module.encoder(torch.zeros_like(data).fill_(0.2))
        pl_module.log("encoder_test", z.mean())
                   
class InputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            trainer.logger.experiment.add_histogram("image_histogram", x, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_histogram", y, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_normalized_histogram", pl_module.embed_normalizer(y), global_step=trainer.global_step)