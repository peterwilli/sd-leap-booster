import os
import subprocess
import argparse
import time

file_path = os.path.abspath(os.path.dirname(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--input_folder", type=str, default=os.path.join(file_path, "lora_dataset"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args(args)

def main():
    args = parse_args()
    image_folders = os.listdir(args.input_folder)
    image_folders.sort()
    for image_folder in image_folders:
        print(f"Processing: {image_folder}")
        output_path = os.path.join(args.input_folder, image_folder, "models")

        models_found_1 = os.path.exists(os.path.join(output_path, "step_1000.safetensors"))
        models_found_2 = os.path.exists(os.path.join(output_path, "step_inv_1000.safetensors"))
        if models_found_1 and models_found_2:
             print("already fully trained")
             continue
        
        image_folder = os.path.join(args.input_folder, image_folder, "images")
        os.makedirs(output_path, exist_ok=True)

        cmd = ['lora_pti',
            '--pretrained_model_name_or_path=' + args.pretrained_model_name_or_path,
            '--instance_data_dir=' + image_folder,
            '--output_dir=' + output_path,
            '--train_text_encoder',
            '--resolution=512',
            '--train_batch_size=1',
            '--gradient_accumulation_steps=4',
            '--scale_lr',
            '--learning_rate_unet=1e-4',
            '--learning_rate_text=1e-5',
            '--learning_rate_ti=5e-4',
            '--color_jitter',
            '--lr_scheduler="linear"',
            '--lr_warmup_steps=0',
            '--placeholder_tokens="<s1>"',
            '--use_template="object"',
            '--save_steps=100',
            '--max_train_steps_ti=1000',
            '--max_train_steps_tuning=1000',
            '--perform_inversion=True',
            '--clip_ti_decay',
            '--weight_decay_ti=0.000',
            '--weight_decay_lora=0.001',
            '--continue_inversion',
            '--continue_inversion_lr=1e-4',
            '--device="cuda:0"',
            '--lora_rank=1']

        subprocess.run(cmd)
        time.sleep(2)

if __name__ == "__main__":
    main()