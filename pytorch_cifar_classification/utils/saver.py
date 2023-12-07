import torch
import os
from datetime import datetime


def save_model(model, model_name, directory='models'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{model_name}_{current_time}.pt"
    path = os.path.join(directory, file_name)

    torch.save(model.state_dict(), path)
