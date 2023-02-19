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
import re
import pytorch_lightning as pl
from lora_diffusion import patch_pipe
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import diffusers
from lora_diffusion import tune_lora_scale, patch_pipe

def parse_args(args=None):
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--images_per_prompt", type=int, default=2)
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

def gen_images(dataset_path, model_id, images_per_prompt, **kwargs):
    model_files = os.listdir(dataset_path)
    random.shuffle(model_files)
    X = None
    sorted_keys = None
    for model_file in tqdm(model_files, desc="Generating images..."):
        model_path = os.path.join(dataset_path, model_file, "models")
        model_file_path = os.path.join(model_path, "step_1000.safetensors")
        if os.path.exists(model_file_path):
            generated_images_path = os.path.join(dataset_path, model_file, "images_generated")
            if os.path.exists(generated_images_path):
                shutil.rmtree(generated_images_path)
            os.makedirs(generated_images_path, exist_ok=True)
            images = generate_from_lora_tensors(model_id, model_file_path, images_per_prompt)
            torch.cuda.empty_cache()
            for i, entry in enumerate(images):
                image = entry['image']
                name = entry['name']
                image.save(os.path.join(generated_images_path, f"{name}_{i}.jpg"))

def main():
    pl.seed_everything(1)
    args = parse_args()
    gen_images(**vars(args))
    
if __name__ == "__main__":
    main()