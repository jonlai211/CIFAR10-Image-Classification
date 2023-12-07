import torch
import os


def count_parameters(model):
    """Calculate the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def get_model_size(model_path):
    """Calculate the size of the model file in MB."""
    return os.path.getsize(model_path) / 1e6


def print_model_info(model, model_path):
    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")

    if os.path.exists(model_path):
        model_size = get_model_size(model_path)
        print(f"Model size: {model_size} MB")
    else:
        print("Model file not found.")
