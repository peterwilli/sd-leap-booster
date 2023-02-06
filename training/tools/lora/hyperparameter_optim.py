import subprocess
import random
import os
import sys
import argparse
import optuna
from safetensors import safe_open
import torch

file_path = os.path.abspath(os.path.dirname(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=os.path.join(file_path, ".hyperparam_optim_tmp"))
    parser.add_argument("--hyperparam_optim_tmp", type=str, default=os.path.join(file_path, ".hyperparam_optim_tmp"))
    parser.add_argument("--leap_model_path", type=str, required=True)
    return parser.parse_args(args)

args = parse_args()

def get_flat_safetensors(path):
    with safe_open(path, framework="pt") as f:
        tensor = None
        keys = list(f.keys())
        # Avoiding undefined behaviour: Making sure we always use keys in alphabethical order!
        keys.sort()
        for k in keys:
            if tensor is None:
                tensor = f.get_tensor(k).flatten()
            else:
                tensor = torch.cat((tensor, f.get_tensor(k).flatten()), 0)
        return tensor

def objective(trial):
    test_concepts = ['vol']
    
    learning_rate_unet = trial.suggest_float('learning_rate_unet', 1e-5, 2e-3)
    learning_rate_text = trial.suggest_float('learning_rate_text', 1e-6, 2e-3)
    continue_inversion_lr = trial.suggest_float('continue_inversion_lr', 1e-5, 2e-3)
    learning_rate_ti = trial.suggest_float('learning_rate_ti', 1e-4, 2e-3)
    default_schedulers = [
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup"
    ]
    lr_scheduler = trial.suggest_categorical('lr_scheduler', default_schedulers)
    lr_scheduler_lora = trial.suggest_categorical('lr_scheduler_lora', default_schedulers)
    color_jitter = trial.suggest_categorical('color_jitter', ["True", "False"])

    tensor_diffs = 0

    for concept in test_concepts:
        instance_data_dir = os.path.abspath(os.path.join(file_path, "..", "..", "test_images", concept))
        cli_path = os.path.abspath(os.path.join(file_path, "..", "..", "..", "bin", "leap_lora"))
        shell_command = f"""
        {sys.executable} -u {cli_path} \
            --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"  \
            --instance_data_dir="{instance_data_dir}" \
            --leap_model_path={args.leap_model_path} \
            --output_dir="{args.output_folder}" \
            --train_text_encoder \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=4 \
            --scale_lr \
            --learning_rate_unet={learning_rate_unet} \
            --learning_rate_text={learning_rate_text} \
            --learning_rate_ti={learning_rate_ti} \
            --color_jitter={color_jitter} \
            --lr_scheduler="{lr_scheduler}" \
            --lr_scheduler_lora="{lr_scheduler_lora}" \
            --lr_warmup_steps=10 \
            --lr_warmup_steps_lora=10 \
            --placeholder_tokens="<s1>" \
            --use_template="object" \
            --save_steps=100 \
            --max_train_steps_ti=100 \
            --max_train_steps_tuning=100 \
            --perform_inversion=True \
            --clip_ti_decay \
            --weight_decay_ti=0.000 \
            --weight_decay_lora=0.001 \
            --continue_inversion \
            --continue_inversion_lr={continue_inversion_lr} \
            --device="cuda:0" \
            --lora_rank=1
        """.strip()
        p = subprocess.Popen(shell_command, shell=True)
        p.communicate()
        
        original_flat = get_flat_safetensors(os.path.join(file_path, "original_loras", f"{concept}.safetensors"))
        generated_flat = get_flat_safetensors(os.path.join(args.output_folder, "step_100.safetensors"))
        print("original_flat", original_flat)
        print("generated_flat", generated_flat)
        tensor_diffs += abs(original_flat - generated_flat).mean().item()
        
    return tensor_diffs / len(test_concepts)

def main():
    study = optuna.create_study(
        storage="sqlite:///optuna.sqlite3",  # Specify the storage URL here.
        study_name="leap_lora",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100)
    print(study.best_params)

if __name__ == "__main__":
    main()
