import pytorch_lightning as pl

class InputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            trainer.logger.experiment.add_histogram("image_histogram", x, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_histogram", y, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_normalized_histogram", pl_module.embed_normalizer(y), global_step=trainer.global_step)