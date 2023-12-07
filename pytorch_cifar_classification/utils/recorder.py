import csv
import os
from datetime import datetime


class Recorder:
    def __init__(self, model_name, mode='train', output_dir='outputs'):
        self.model_name = model_name
        self.mode = mode
        self.output_dir = output_dir
        self.all_metrics = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_name = f"{self.mode}_{self.model_name}_{current_time}.csv"

    def update_metrics(self, epoch, metrics):
        self.all_metrics.append(metrics)
        metrics['epoch'] = epoch

    def save(self):
        file_path = os.path.join(self.output_dir, self.file_name)
        with open(file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.all_metrics)
