import argparse
import os
from leap_sd import LM, Autoencoder
from tqdm import tqdm
import torch
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import safe_open as open_safetensors
import numpy as np
from time import time
import math
import unicodedata
import random
import shutil
import tempfile
import re
import pytorch_lightning as pl
from lora_diffusion import patch_pipe
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import diffusers
from lora_diffusion import tune_lora_scale, patch_pipe

def parse_args(args=None):
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--images_per_prompt", type=int, default=2)
    parser.add_argument("--num_models", type=int, default=1000)
    parser.add_argument("--output_folder", type=str, default=os.path.join(file_path, "random_embeds"))
    parser.add_argument("--leap_checkpoint", type=str, required=True)
    return parser.parse_args(args)

def generate_from_lora_tensors(model_id, tensors_path, images_per_prompt):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        "cuda"
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe._progress_bar_config = {'disable': True}
    patch_pipe(
        pipe,
        tensors_path,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )
    tune_lora_scale(pipe.unet, 0.1)
    tune_lora_scale(pipe.text_encoder, 0.1)
    imagenet_templates_small = [
        "{}, realistic photo",
        "{}, realistic render",
        "{}, painting",
        "{}, anime",
        "{}, greg ruthkowski",
        "{}, cartoon",
        "{}, vector art",
        "{}, clip art"
    ]
    images = []
    gen_name = tensors_path.split("/")[-3]
    for image_idx in tqdm(range(images_per_prompt), desc=f"Rendering {gen_name}", leave=False):
        for text in (pbar := tqdm(imagenet_templates_small, leave=False)):
            text = text.format("<s1>")
            pbar.set_description(f"Rendering {text}")
            image = pipe(text, num_inference_steps=25, guidance_scale=9).images[0]
            images.append({ 'name': slugify(text), 'image': image })
    return images

def slugify(value):
    """
    Converts to lowercase, removes non-word characters (alphanumerics and
    underscores) and converts spaces to hyphens. Also strips leading and
    trailing whitespace.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '-', value)

def gen_images(leap, model_id, images_per_prompt, num_models, output_folder, **kwargs):
    num_dimensions = leap.pca['pca'].n_components
    metadata_obj = {'<s1>': '<embed>', 'text_encoder': '["CLIPAttention"]', 'text_encoder:0:rank': '1', 'text_encoder:10:rank': '1', 'text_encoder:11:rank': '1', 'text_encoder:12:rank': '1', 'text_encoder:13:rank': '1', 'text_encoder:14:rank': '1', 'text_encoder:15:rank': '1', 'text_encoder:16:rank': '1', 'text_encoder:17:rank': '1', 'text_encoder:18:rank': '1', 'text_encoder:19:rank': '1', 'text_encoder:1:rank': '1', 'text_encoder:20:rank': '1', 'text_encoder:21:rank': '1', 'text_encoder:22:rank': '1', 'text_encoder:23:rank': '1', 'text_encoder:24:rank': '1', 'text_encoder:25:rank': '1', 'text_encoder:26:rank': '1', 'text_encoder:27:rank': '1', 'text_encoder:28:rank': '1', 'text_encoder:29:rank': '1', 'text_encoder:2:rank': '1', 'text_encoder:30:rank': '1', 'text_encoder:31:rank': '1', 'text_encoder:32:rank': '1', 'text_encoder:33:rank': '1', 'text_encoder:34:rank': '1', 'text_encoder:35:rank': '1', 'text_encoder:36:rank': '1', 'text_encoder:37:rank': '1', 'text_encoder:38:rank': '1', 'text_encoder:39:rank': '1', 'text_encoder:3:rank': '1', 'text_encoder:40:rank': '1', 'text_encoder:41:rank': '1', 'text_encoder:42:rank': '1', 'text_encoder:43:rank': '1', 'text_encoder:44:rank': '1', 'text_encoder:45:rank': '1', 'text_encoder:46:rank': '1', 'text_encoder:47:rank': '1', 'text_encoder:48:rank': '1', 'text_encoder:49:rank': '1', 'text_encoder:4:rank': '1', 'text_encoder:50:rank': '1', 'text_encoder:51:rank': '1', 'text_encoder:52:rank': '1', 'text_encoder:53:rank': '1', 'text_encoder:54:rank': '1', 'text_encoder:55:rank': '1', 'text_encoder:56:rank': '1', 'text_encoder:57:rank': '1', 'text_encoder:58:rank': '1', 'text_encoder:59:rank': '1', 'text_encoder:5:rank': '1', 'text_encoder:60:rank': '1', 'text_encoder:61:rank': '1', 'text_encoder:62:rank': '1', 'text_encoder:63:rank': '1', 'text_encoder:64:rank': '1', 'text_encoder:65:rank': '1', 'text_encoder:66:rank': '1', 'text_encoder:67:rank': '1', 'text_encoder:68:rank': '1', 'text_encoder:69:rank': '1', 'text_encoder:6:rank': '1', 'text_encoder:70:rank': '1', 'text_encoder:71:rank': '1', 'text_encoder:72:rank': '1', 'text_encoder:73:rank': '1', 'text_encoder:74:rank': '1', 'text_encoder:75:rank': '1', 'text_encoder:76:rank': '1', 'text_encoder:77:rank': '1', 'text_encoder:78:rank': '1', 'text_encoder:79:rank': '1', 'text_encoder:7:rank': '1', 'text_encoder:80:rank': '1', 'text_encoder:81:rank': '1', 'text_encoder:82:rank': '1', 'text_encoder:83:rank': '1', 'text_encoder:84:rank': '1', 'text_encoder:85:rank': '1', 'text_encoder:86:rank': '1', 'text_encoder:87:rank': '1', 'text_encoder:88:rank': '1', 'text_encoder:89:rank': '1', 'text_encoder:8:rank': '1', 'text_encoder:90:rank': '1', 'text_encoder:91:rank': '1', 'text_encoder:9:rank': '1', 'unet': '["Attention", "GEGLU", "CrossAttention"]', 'unet:0:rank': '1', 'unet:100:rank': '1', 'unet:101:rank': '1', 'unet:102:rank': '1', 'unet:103:rank': '1', 'unet:104:rank': '1', 'unet:105:rank': '1', 'unet:106:rank': '1', 'unet:107:rank': '1', 'unet:108:rank': '1', 'unet:109:rank': '1', 'unet:10:rank': '1', 'unet:110:rank': '1', 'unet:111:rank': '1', 'unet:112:rank': '1', 'unet:113:rank': '1', 'unet:114:rank': '1', 'unet:115:rank': '1', 'unet:116:rank': '1', 'unet:117:rank': '1', 'unet:118:rank': '1', 'unet:119:rank': '1', 'unet:11:rank': '1', 'unet:120:rank': '1', 'unet:121:rank': '1', 'unet:122:rank': '1', 'unet:123:rank': '1', 'unet:124:rank': '1', 'unet:125:rank': '1', 'unet:126:rank': '1', 'unet:127:rank': '1', 'unet:128:rank': '1', 'unet:129:rank': '1', 'unet:12:rank': '1', 'unet:130:rank': '1', 'unet:131:rank': '1', 'unet:132:rank': '1', 'unet:133:rank': '1', 'unet:134:rank': '1', 'unet:135:rank': '1', 'unet:136:rank': '1', 'unet:137:rank': '1', 'unet:138:rank': '1', 'unet:139:rank': '1', 'unet:13:rank': '1', 'unet:140:rank': '1', 'unet:141:rank': '1', 'unet:142:rank': '1', 'unet:143:rank': '1', 'unet:14:rank': '1', 'unet:15:rank': '1', 'unet:16:rank': '1', 'unet:17:rank': '1', 'unet:18:rank': '1', 'unet:19:rank': '1', 'unet:1:rank': '1', 'unet:20:rank': '1', 'unet:21:rank': '1', 'unet:22:rank': '1', 'unet:23:rank': '1', 'unet:24:rank': '1', 'unet:25:rank': '1', 'unet:26:rank': '1', 'unet:27:rank': '1', 'unet:28:rank': '1', 'unet:29:rank': '1', 'unet:2:rank': '1', 'unet:30:rank': '1', 'unet:31:rank': '1', 'unet:32:rank': '1', 'unet:33:rank': '1', 'unet:34:rank': '1', 'unet:35:rank': '1', 'unet:36:rank': '1', 'unet:37:rank': '1', 'unet:38:rank': '1', 'unet:39:rank': '1', 'unet:3:rank': '1', 'unet:40:rank': '1', 'unet:41:rank': '1', 'unet:42:rank': '1', 'unet:43:rank': '1', 'unet:44:rank': '1', 'unet:45:rank': '1', 'unet:46:rank': '1', 'unet:47:rank': '1', 'unet:48:rank': '1', 'unet:49:rank': '1', 'unet:4:rank': '1', 'unet:50:rank': '1', 'unet:51:rank': '1', 'unet:52:rank': '1', 'unet:53:rank': '1', 'unet:54:rank': '1', 'unet:55:rank': '1', 'unet:56:rank': '1', 'unet:57:rank': '1', 'unet:58:rank': '1', 'unet:59:rank': '1', 'unet:5:rank': '1', 'unet:60:rank': '1', 'unet:61:rank': '1', 'unet:62:rank': '1', 'unet:63:rank': '1', 'unet:64:rank': '1', 'unet:65:rank': '1', 'unet:66:rank': '1', 'unet:67:rank': '1', 'unet:68:rank': '1', 'unet:69:rank': '1', 'unet:6:rank': '1', 'unet:70:rank': '1', 'unet:71:rank': '1', 'unet:72:rank': '1', 'unet:73:rank': '1', 'unet:74:rank': '1', 'unet:75:rank': '1', 'unet:76:rank': '1', 'unet:77:rank': '1', 'unet:78:rank': '1', 'unet:79:rank': '1', 'unet:7:rank': '1', 'unet:80:rank': '1', 'unet:81:rank': '1', 'unet:82:rank': '1', 'unet:83:rank': '1', 'unet:84:rank': '1', 'unet:85:rank': '1', 'unet:86:rank': '1', 'unet:87:rank': '1', 'unet:88:rank': '1', 'unet:89:rank': '1', 'unet:8:rank': '1', 'unet:90:rank': '1', 'unet:91:rank': '1', 'unet:92:rank': '1', 'unet:93:rank': '1', 'unet:94:rank': '1', 'unet:95:rank': '1', 'unet:96:rank': '1', 'unet:97:rank': '1', 'unet:98:rank': '1', 'unet:99:rank': '1', 'unet:9:rank': '1'}
    for i in range(num_models):
        torch.cuda.empty_cache()
        noise = torch.zeros(num_dimensions).uniform_(0, 1)
        boosted_embed = leap.post_process(noise)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tensors_path = os.path.join(tmpdirname, "noised_tensors.safetensors")
            save_safetensors(boosted_embed, tensors_path, metadata_obj)
            images = generate_from_lora_tensors(model_id, tensors_path, images_per_prompt)
            output_folder_real = os.path.join(output_folder, f"random_{i}")
            if os.path.exists(output_folder_real):
                shutil.rmtree(output_folder_real)
            os.makedirs(output_folder_real, exist_ok=True)
            for idx, entry in enumerate(images):
                image = entry['image']
                name = entry['name']
                image.save(os.path.join(output_folder_real, f"{name}_{idx}.jpg"))

def init_model(leap_checkpoint, **kwargs):
    leap = LM.load_from_checkpoint(leap_checkpoint)
    leap.eval()
    return leap

def main():
    pl.seed_everything(1)
    args = parse_args()
    args.leap = init_model(**vars(args))
    gen_images(**vars(args))
    
if __name__ == "__main__":
    main()