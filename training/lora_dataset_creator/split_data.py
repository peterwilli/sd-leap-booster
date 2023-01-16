import os
import subprocess
import argparse
import time
import random
import shutil

file_path = os.path.abspath(os.path.dirname(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=os.path.join(file_path, "lora_dataset"))
    parser.add_argument("--val_amount", type=int, default=4)
    return parser.parse_args(args)

def move_files(input_folder, files, type: str):
    target_path = os.path.join(input_folder, type)
    for file in files:
        source_folder = os.path.join(input_folder, file)
        target_folder = os.path.join(target_path, file)
        shutil.move(source_folder, target_folder)

def main():
    args = parse_args()
    files = os.listdir(args.input_folder)
    if "val" in files and "train" in files:
        print("Already splitted up!")
        return
    random.shuffle(files)
    val_files = files[:args.val_amount]
    train_files = files[args.val_amount:]
    move_files(args.input_folder, val_files, "val")
    move_files(args.input_folder, train_files, "train")

if __name__ == "__main__":
    main()