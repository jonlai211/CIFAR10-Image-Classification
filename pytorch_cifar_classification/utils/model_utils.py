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


def estimate_model_memory_usage(model, input_size, dtype=torch.float32):
    """Estimate the model's memory usage in MB during training."""
    param_size = sum(p.numel() for p in model.parameters()) * dtype.itemsize
    forward_backward_size = param_size * 2  # Forward and backward pass
    input_size = torch.zeros(input_size).size().numel() * dtype.itemsize
    total_size = (param_size + forward_backward_size + input_size) / 1e6  # Convert to MB
    return total_size


def print_model_memory_info(model, input_size):
    memory_size = estimate_model_memory_usage(model, input_size)
    print(f"Estimated memory usage: {memory_size} MB")
