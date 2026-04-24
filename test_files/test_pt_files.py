import torch
import os

def print_structure(files_path):
    data = torch.load(files_path, map_location="cpu")
    print(f"Type: {type(data)}")
    if isinstance(data, torch.Tensor):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"Key: {key} | Shape: {value.shape} | Dtype: {value.dtype}")
            else:
                print(f"Key: {key} | Type: {type(value)}")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            if isinstance(item, torch.Tensor):
                print(f"[{i}] Shape: {item.shape} | Dtype: {item.dtype}")
            else:
                print(f"[{i}] Type: {type(item)}")
    else:
        print(f"Unsupported type: {type(data)}")

if __name__ == "__main__":
    files_path = "/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training/cache/sample_000000"
    pt_files = [ file for file in os.listdir(files_path) if file.endswith('.pt') ]
    for pt_file in pt_files:
        print(f"\nInspecting file: {pt_file}")
        print_structure(os.path.join(files_path, pt_file))
