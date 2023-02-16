from datamodule import ImageWeightsModule
import pytorch_lightning as pl
import torch
from leap_sd import EmbedAutoencoder
import argparse
import os
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor
from callbacks import GenerateCallback
from dataset_utils import get_extrema

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--linear_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--scaling", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2048)
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)

@torch.no_grad()
def init_extrema(args, dm):
    print("Getting extrema")
    extrema = get_extrema(dm.train_dataloader(), args.mapping)
    print(f"Extrema of entire training set: {extrema}")
    return extrema

def main():
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1)
    
    args = parse_args()
    dm = ImageWeightsModule(args.dataset_path, args.batch_size, augment_training=False)
    for _, y in dm.train_dataloader():
        args.size_in = y.shape[1]
        break
    mapping = {'<s1>': {'len': 1024, 'shape': [1024]}, 'text_encoder:0:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:0:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:10:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:10:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:11:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:11:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:12:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:12:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:13:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:13:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:14:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:14:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:15:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:15:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:16:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:16:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:17:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:17:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:18:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:18:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:19:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:19:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:1:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:1:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:20:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:20:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:21:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:21:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:22:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:22:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:23:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:23:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:24:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:24:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:25:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:25:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:26:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:26:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:27:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:27:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:28:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:28:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:29:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:29:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:2:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:2:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:30:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:30:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:31:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:31:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:32:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:32:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:33:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:33:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:34:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:34:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:35:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:35:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:36:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:36:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:37:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:37:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:38:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:38:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:39:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:39:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:3:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:3:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:40:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:40:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:41:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:41:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:42:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:42:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:43:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:43:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:44:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:44:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:45:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:45:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:46:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:46:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:47:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:47:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:48:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:48:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:49:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:49:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:4:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:4:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:50:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:50:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:51:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:51:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:52:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:52:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:53:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:53:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:54:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:54:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:55:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:55:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:56:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:56:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:57:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:57:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:58:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:58:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:59:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:59:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:5:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:5:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:60:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:60:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:61:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:61:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:62:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:62:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:63:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:63:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:64:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:64:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:65:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:65:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:66:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:66:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:67:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:67:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:68:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:68:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:69:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:69:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:6:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:6:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:70:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:70:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:71:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:71:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:72:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:72:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:73:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:73:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:74:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:74:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:75:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:75:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:76:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:76:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:77:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:77:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:78:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:78:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:79:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:79:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:7:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:7:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:80:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:80:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:81:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:81:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:82:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:82:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:83:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:83:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:84:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:84:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:85:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:85:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:86:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:86:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:87:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:87:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:88:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:88:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:89:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:89:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:8:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:8:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:90:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:90:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:91:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:91:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:9:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:9:up': {'len': 1024, 'shape': [1024, 1]}, 'unet:0:down': {'len': 320, 'shape': [1, 320]}, 'unet:0:up': {'len': 320, 'shape': [320, 1]}, 'unet:100:down': {'len': 640, 'shape': [1, 640]}, 'unet:100:up': {'len': 640, 'shape': [640, 1]}, 'unet:101:down': {'len': 640, 'shape': [1, 640]}, 'unet:101:up': {'len': 640, 'shape': [640, 1]}, 'unet:102:down': {'len': 640, 'shape': [1, 640]}, 'unet:102:up': {'len': 640, 'shape': [640, 1]}, 'unet:103:down': {'len': 640, 'shape': [1, 640]}, 'unet:103:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:104:down': {'len': 640, 'shape': [1, 640]}, 'unet:104:up': {'len': 640, 'shape': [640, 1]}, 'unet:105:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:105:up': {'len': 640, 'shape': [640, 1]}, 'unet:106:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:106:up': {'len': 640, 'shape': [640, 1]}, 'unet:107:down': {'len': 640, 'shape': [1, 640]}, 'unet:107:up': {'len': 640, 'shape': [640, 1]}, 'unet:108:down': {'len': 320, 'shape': [1, 320]}, 'unet:108:up': {'len': 320, 'shape': [320, 1]}, 'unet:109:down': {'len': 320, 'shape': [1, 320]}, 'unet:109:up': {'len': 320, 'shape': [320, 1]}, 'unet:10:down': {'len': 320, 'shape': [1, 320]}, 'unet:10:up': {'len': 320, 'shape': [320, 1]}, 'unet:110:down': {'len': 320, 'shape': [1, 320]}, 'unet:110:up': {'len': 320, 'shape': [320, 1]}, 'unet:111:down': {'len': 320, 'shape': [1, 320]}, 'unet:111:up': {'len': 320, 'shape': [320, 1]}, 'unet:112:down': {'len': 320, 'shape': [1, 320]}, 'unet:112:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:113:down': {'len': 320, 'shape': [1, 320]}, 'unet:113:up': {'len': 320, 'shape': [320, 1]}, 'unet:114:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:114:up': {'len': 320, 'shape': [320, 1]}, 'unet:115:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:115:up': {'len': 320, 'shape': [320, 1]}, 'unet:116:down': {'len': 320, 'shape': [1, 320]}, 'unet:116:up': {'len': 320, 'shape': [320, 1]}, 'unet:117:down': {'len': 320, 'shape': [1, 320]}, 'unet:117:up': {'len': 320, 'shape': [320, 1]}, 'unet:118:down': {'len': 320, 'shape': [1, 320]}, 'unet:118:up': {'len': 320, 'shape': [320, 1]}, 'unet:119:down': {'len': 320, 'shape': [1, 320]}, 'unet:119:up': {'len': 320, 'shape': [320, 1]}, 'unet:11:down': {'len': 320, 'shape': [1, 320]}, 'unet:11:up': {'len': 320, 'shape': [320, 1]}, 'unet:120:down': {'len': 320, 'shape': [1, 320]}, 'unet:120:up': {'len': 320, 'shape': [320, 1]}, 'unet:121:down': {'len': 320, 'shape': [1, 320]}, 'unet:121:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:122:down': {'len': 320, 'shape': [1, 320]}, 'unet:122:up': {'len': 320, 'shape': [320, 1]}, 'unet:123:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:123:up': {'len': 320, 'shape': [320, 1]}, 'unet:124:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:124:up': {'len': 320, 'shape': [320, 1]}, 'unet:125:down': {'len': 320, 'shape': [1, 320]}, 'unet:125:up': {'len': 320, 'shape': [320, 1]}, 'unet:126:down': {'len': 320, 'shape': [1, 320]}, 'unet:126:up': {'len': 320, 'shape': [320, 1]}, 'unet:127:down': {'len': 320, 'shape': [1, 320]}, 'unet:127:up': {'len': 320, 'shape': [320, 1]}, 'unet:128:down': {'len': 320, 'shape': [1, 320]}, 'unet:128:up': {'len': 320, 'shape': [320, 1]}, 'unet:129:down': {'len': 320, 'shape': [1, 320]}, 'unet:129:up': {'len': 320, 'shape': [320, 1]}, 'unet:12:down': {'len': 320, 'shape': [1, 320]}, 'unet:12:up': {'len': 320, 'shape': [320, 1]}, 'unet:130:down': {'len': 320, 'shape': [1, 320]}, 'unet:130:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:131:down': {'len': 320, 'shape': [1, 320]}, 'unet:131:up': {'len': 320, 'shape': [320, 1]}, 'unet:132:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:132:up': {'len': 320, 'shape': [320, 1]}, 'unet:133:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:133:up': {'len': 320, 'shape': [320, 1]}, 'unet:134:down': {'len': 320, 'shape': [1, 320]}, 'unet:134:up': {'len': 320, 'shape': [320, 1]}, 'unet:135:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:135:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:136:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:136:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:137:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:137:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:138:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:138:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:139:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:139:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:13:down': {'len': 320, 'shape': [1, 320]}, 'unet:13:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:140:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:140:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:141:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:141:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:142:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:142:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:143:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:143:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:14:down': {'len': 320, 'shape': [1, 320]}, 'unet:14:up': {'len': 320, 'shape': [320, 1]}, 'unet:15:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:15:up': {'len': 320, 'shape': [320, 1]}, 'unet:16:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:16:up': {'len': 320, 'shape': [320, 1]}, 'unet:17:down': {'len': 320, 'shape': [1, 320]}, 'unet:17:up': {'len': 320, 'shape': [320, 1]}, 'unet:18:down': {'len': 640, 'shape': [1, 640]}, 'unet:18:up': {'len': 640, 'shape': [640, 1]}, 'unet:19:down': {'len': 640, 'shape': [1, 640]}, 'unet:19:up': {'len': 640, 'shape': [640, 1]}, 'unet:1:down': {'len': 320, 'shape': [1, 320]}, 'unet:1:up': {'len': 320, 'shape': [320, 1]}, 'unet:20:down': {'len': 640, 'shape': [1, 640]}, 'unet:20:up': {'len': 640, 'shape': [640, 1]}, 'unet:21:down': {'len': 640, 'shape': [1, 640]}, 'unet:21:up': {'len': 640, 'shape': [640, 1]}, 'unet:22:down': {'len': 640, 'shape': [1, 640]}, 'unet:22:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:23:down': {'len': 640, 'shape': [1, 640]}, 'unet:23:up': {'len': 640, 'shape': [640, 1]}, 'unet:24:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:24:up': {'len': 640, 'shape': [640, 1]}, 'unet:25:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:25:up': {'len': 640, 'shape': [640, 1]}, 'unet:26:down': {'len': 640, 'shape': [1, 640]}, 'unet:26:up': {'len': 640, 'shape': [640, 1]}, 'unet:27:down': {'len': 640, 'shape': [1, 640]}, 'unet:27:up': {'len': 640, 'shape': [640, 1]}, 'unet:28:down': {'len': 640, 'shape': [1, 640]}, 'unet:28:up': {'len': 640, 'shape': [640, 1]}, 'unet:29:down': {'len': 640, 'shape': [1, 640]}, 'unet:29:up': {'len': 640, 'shape': [640, 1]}, 'unet:2:down': {'len': 320, 'shape': [1, 320]}, 'unet:2:up': {'len': 320, 'shape': [320, 1]}, 'unet:30:down': {'len': 640, 'shape': [1, 640]}, 'unet:30:up': {'len': 640, 'shape': [640, 1]}, 'unet:31:down': {'len': 640, 'shape': [1, 640]}, 'unet:31:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:32:down': {'len': 640, 'shape': [1, 640]}, 'unet:32:up': {'len': 640, 'shape': [640, 1]}, 'unet:33:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:33:up': {'len': 640, 'shape': [640, 1]}, 'unet:34:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:34:up': {'len': 640, 'shape': [640, 1]}, 'unet:35:down': {'len': 640, 'shape': [1, 640]}, 'unet:35:up': {'len': 640, 'shape': [640, 1]}, 'unet:36:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:36:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:37:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:37:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:38:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:38:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:39:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:39:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:3:down': {'len': 320, 'shape': [1, 320]}, 'unet:3:up': {'len': 320, 'shape': [320, 1]}, 'unet:40:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:40:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:41:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:41:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:42:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:42:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:43:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:43:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:44:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:44:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:45:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:45:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:46:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:46:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:47:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:47:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:48:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:48:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:49:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:49:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:4:down': {'len': 320, 'shape': [1, 320]}, 'unet:4:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:50:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:50:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:51:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:51:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:52:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:52:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:53:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:53:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:54:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:54:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:55:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:55:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:56:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:56:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:57:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:57:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:58:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:58:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:59:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:59:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:5:down': {'len': 320, 'shape': [1, 320]}, 'unet:5:up': {'len': 320, 'shape': [320, 1]}, 'unet:60:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:60:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:61:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:61:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:62:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:62:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:63:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:63:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:64:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:64:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:65:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:65:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:66:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:66:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:67:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:67:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:68:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:68:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:69:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:69:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:6:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:6:up': {'len': 320, 'shape': [320, 1]}, 'unet:70:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:70:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:71:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:71:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:72:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:72:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:73:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:73:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:74:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:74:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:75:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:75:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:76:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:76:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:77:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:77:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:78:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:78:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:79:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:79:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:7:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:7:up': {'len': 320, 'shape': [320, 1]}, 'unet:80:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:80:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:81:down': {'len': 640, 'shape': [1, 640]}, 'unet:81:up': {'len': 640, 'shape': [640, 1]}, 'unet:82:down': {'len': 640, 'shape': [1, 640]}, 'unet:82:up': {'len': 640, 'shape': [640, 1]}, 'unet:83:down': {'len': 640, 'shape': [1, 640]}, 'unet:83:up': {'len': 640, 'shape': [640, 1]}, 'unet:84:down': {'len': 640, 'shape': [1, 640]}, 'unet:84:up': {'len': 640, 'shape': [640, 1]}, 'unet:85:down': {'len': 640, 'shape': [1, 640]}, 'unet:85:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:86:down': {'len': 640, 'shape': [1, 640]}, 'unet:86:up': {'len': 640, 'shape': [640, 1]}, 'unet:87:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:87:up': {'len': 640, 'shape': [640, 1]}, 'unet:88:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:88:up': {'len': 640, 'shape': [640, 1]}, 'unet:89:down': {'len': 640, 'shape': [1, 640]}, 'unet:89:up': {'len': 640, 'shape': [640, 1]}, 'unet:8:down': {'len': 320, 'shape': [1, 320]}, 'unet:8:up': {'len': 320, 'shape': [320, 1]}, 'unet:90:down': {'len': 640, 'shape': [1, 640]}, 'unet:90:up': {'len': 640, 'shape': [640, 1]}, 'unet:91:down': {'len': 640, 'shape': [1, 640]}, 'unet:91:up': {'len': 640, 'shape': [640, 1]}, 'unet:92:down': {'len': 640, 'shape': [1, 640]}, 'unet:92:up': {'len': 640, 'shape': [640, 1]}, 'unet:93:down': {'len': 640, 'shape': [1, 640]}, 'unet:93:up': {'len': 640, 'shape': [640, 1]}, 'unet:94:down': {'len': 640, 'shape': [1, 640]}, 'unet:94:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:95:down': {'len': 640, 'shape': [1, 640]}, 'unet:95:up': {'len': 640, 'shape': [640, 1]}, 'unet:96:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:96:up': {'len': 640, 'shape': [640, 1]}, 'unet:97:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:97:up': {'len': 640, 'shape': [640, 1]}, 'unet:98:down': {'len': 640, 'shape': [1, 640]}, 'unet:98:up': {'len': 640, 'shape': [640, 1]}, 'unet:99:down': {'len': 640, 'shape': [1, 640]}, 'unet:99:up': {'len': 640, 'shape': [640, 1]}, 'unet:9:down': {'len': 320, 'shape': [1, 320]}, 'unet:9:up': {'len': 320, 'shape': [320, 1]}}
    args.mapping = mapping
    args.extrema = init_extrema(args, dm)
    all_data_loader = ImageWeightsModule(args.dataset_path, 1, augment_training=False, val_split=0).train_dataloader()
    args.quantity = len(all_data_loader)
    print("args.quantity", args.quantity)
    del all_data_loader
    ae = EmbedAutoencoder(**vars(args))
    ae.train()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    args.callbacks = [
        lr_monitor
    ]

    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(ae, dm)

if __name__ == "__main__":
    main()