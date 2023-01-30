import subprocess
import random
import os
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--output_folder", type=str, default=os.path.join(file_path, "lr_search_result"))
    parser.add_argument("--leap_model_path", type=str, required=True)
    return parser.parse_args(args)

def main():
    args = parse_args()
    test_concepts = ['vol', 'peter_tootsy', 'spongebob']
    file_path = os.path.abspath(os.path.dirname(__file__))
    for idx in range(0, 1000):
        learning_rate_unet = random.uniform(1e-4, 2e-3)
        learning_rate_text = random.uniform(1e-5, 2e-3)
        continue_inversion_lr = random.uniform(1e-4, 2e-3)
        learning_rate_ti = random.uniform(1e-3, 2e-3)
        color_jitter = None
        if random.uniform(0, 1) > 0.5:
            color_jitter = "False"
        else:
            color_jitter = "True"
        for concept in test_concepts:
            output_dir = os.path.join(args.output_folder, f"{concept}_{idx}")
            instance_data_dir = os.path.abspath(os.path.join(file_path, "..", "..", "test_images", concept))
            shell_command = f"""
            leap_lora \
                --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"  \
                --instance_data_dir="{instance_data_dir}" \
                --leap_model_path={args.leap_model_path} \
                --output_dir="{output_dir}" \
                --train_text_encoder \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=4 \
                --scale_lr \
                --learning_rate_unet={learning_rate_unet} \
                --learning_rate_text={learning_rate_text} \
                --learning_rate_ti={learning_rate_ti} \
                --color_jitter \
                --lr_scheduler="constant_with_warmup" \
                --lr_scheduler_lora="constant_with_warmup" \
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
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "command.sh"), 'w') as f:
                f.write(shell_command)
            p = subprocess.Popen(shell_command, stdout=subprocess.PIPE, shell=True)
            p.communicate()

if __name__ == "__main__":
    main()
