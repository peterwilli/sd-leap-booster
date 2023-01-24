import os

def main():
    file_path = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(file_path, "lora_dataset")
    for folder in os.listdir(dataset_path):
        model_folder = os.path.join(dataset_path, folder, "models")
        models_found_1 = os.path.exists(os.path.join(model_folder, "step_1000.safetensors"))
        models_found_2 = os.path.exists(os.path.join(model_folder, "step_inv_1000.safetensors"))
        if not (models_found_1 and models_found_2):
            print(os.path.join(dataset_path, folder), "incomplete")

if __name__ == "__main__":
    main()