from datamodule import ImageWeightsModule, FakeWeightsModule
import pytorch_lightning as pl
import torch
from leap_sd import Autoencoder
import argparse
import os
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)

class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)

def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    for imgs, _ in tqdm(data_loader, desc="Encoding images", leave=False):
        with torch.no_grad():
            z = model.encoder(imgs.to(model.device))
        img_list.append(imgs)
        embed_list.append(z)
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))

def write_embeds_to_tensorboard():
    NUM_IMGS = len(test_set)
    writer = SummaryWriter("tensorboard/")
    writer.add_embedding(
        test_img_embeds[1][:NUM_IMGS],  # Encodings per image
        metadata=[test_set[i][1] for i in range(NUM_IMGS)],  # Adding the labels per image to the plot
        label_img=(test_img_embeds[0][:NUM_IMGS] + 1) / 2.0,
    )  # Adding the original images to the plot

def main():
    args = parse_args()
    dm = ImageWeightsModule(args.dataset_path, args.batch_size)
    ae = Autoencoder(base_channel_size=32, latent_dim=128)
    ae.train()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    args.callbacks = [
        lr_monitor,
        GenerateCallback(get_train_images_part(dm.val_dataloader()), every_n_epochs=1)
    ]

    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(ae, dm)

if __name__ == "__main__":
    main()