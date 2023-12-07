import logging
import os
from datetime import datetime


class Logger:
    def __init__(self, model_name, mode='train', level=logging.INFO):
        self.logger = logging.getLogger(f'{model_name}_{mode}')
        self.logger.setLevel(level)

        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{mode}_{model_name}_{current_time}.log")

        # create file handler which logs even debug messages
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_metrics(self, epoch, metrics):
        message = f"Epoch: {epoch}\n"
        for key, value in metrics.items():
            message += f"{key}: {value}\n"
        self.logger.info(message)

    def log_message(self, message, level=logging.INFO):
        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)
        else:
            self.logger.info(message)
