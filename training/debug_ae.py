import argparse
import os
from leap_sd import LM, Autoencoder
from raw_images_datamodule import ImagesModule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_path", type=str, default="./autoencoder.ckpt")
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    return parser.parse_args(args)

def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    for imgs in tqdm(data_loader, desc="Encoding images", leave=False):
        with torch.no_grad():
            z = model.encoder(imgs.to(model.device))
        img_list.append(imgs)
        embed_list.append(z)
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))

def write_embed(writer, embeds):
    writer.add_embedding(
        embeds[1], 
        label_img=(embeds[0] + 1) / 2.0,
    )

def main():
    args = parse_args()
    ae = Autoencoder.load_from_checkpoint(args.autoencoder_path)
    dm = ImagesModule(args.dataset_path, 10, augment_training=False, val_split=0)
    writer = SummaryWriter("embeds_tensorboard/")
    train_img_embeds = embed_imgs(ae, dm.train_dataloader())
    write_embed(writer, train_img_embeds)
    writer.close()
    
if __name__ == "__main__":
    main()