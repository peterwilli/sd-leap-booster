from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import torch
import gc
import os
import nltk
import random
import argparse
from nltk.corpus import stopwords

file_path = os.path.abspath(os.path.dirname(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--output_folder", type=str, default=os.path.join(file_path, "sd_extracted"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args(args)

imagenet_templates_small = [
    "a photo of a {}",
    "a photo of a big {}",
    "a photo of a small {}",
    "{}, realistic photo",
    "{}, realistic render",
    "{}, realistic painting",
    "{}, cartoon",
    "{}, vector art",
    "{}, clip art"
]

if __name__ == "__main__":
    args = parse_args()
    nltk.download('stopwords')
    model_id_or_path = args.pretrained_model_name_or_path
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id_or_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id_or_path, subfolder="text_encoder"
    )
    token_embeds = text_encoder.get_input_embeddings().weight.data
    def dummy(images, **kwargs):
        return images, False
    # pipeline.safety_checker = dummy
    pipeline = pipeline.to("cuda")
    stopwords_english = stopwords.words('english')
    tokens_to_search = []

    common_english_words = {}
    with open(os.path.join(file_path, "bip39.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                common_english_words[line.strip()] = True

    for token_id in range(token_embeds.shape[0]):
        token_name = tokenizer.decode(token_id)
        if len(token_name) > 3 and token_name.isalnum() and not token_name in stopwords_english and token_name in common_english_words:
            tokens_to_search.append(token_name)

    random.seed(80085)
    random.shuffle(tokens_to_search)

    for token_idx, token_name in enumerate(tokens_to_search):
        token_id = tokenizer.encode(token_name, add_special_tokens=False)
        if len(token_id) > 1:
            raise "Need single token!"
        token_type = "train"
        if len(tokens_to_search) - token_idx <= 4:
            token_type = "val"
        image_output_folder = os.path.join(args.output_folder, token_type, token_name)
        if os.path.exists(image_output_folder):
            print(f"Skipping {image_output_folder} because it already exists")
            continue
        learned_embeds = token_embeds[token_id][0]
        concept_images_folder = os.path.join(image_output_folder, 'concept_images')
        os.makedirs(concept_images_folder, exist_ok = True)
        learned_embeds_dict = {token_name: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(image_output_folder, "learned_embeds.bin"))
        images_per_side = 4
        for image_idx in range(images_per_side):
            text = random.choice(imagenet_templates_small).format(token_name)
            print(f"Doing {token_name} with prompt: '{text}'...")
            image = pipeline(
                text,
                num_inference_steps=50,
                guidance_scale=9,
                width=args.width,
                height=args.height
            ).images[0]
            image.save(os.path.join(concept_images_folder, f"image_{image_idx}.png"))
