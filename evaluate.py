import torch
import torch.nn as nn
from pytorch_cifar_classification.datasets.data_loader_manager import DataLoaderManager
from pytorch_cifar_classification.models.origin_net import Net
from pytorch_cifar_classification.utils.metrics import compute_accuracy, compute_loss, compute_precision_recall_fscore
from pytorch_cifar_classification.utils.logger import Logger
from pytorch_cifar_classification.utils.recorder import Recorder
from pytorch_cifar_classification.utils.predictor import make_predictions, map_class_names, class_statistics, \
    calculate_accuracy
import torchvision.models as models


def evaluate():
    model_name = "ResNet18"

    logger = Logger(model_name, mode='evaluate')
    logger.log_message("Start evaluation ...")
    recorder = Recorder(model_name, mode='evaluate')

    # Initialize DataLoaderManager for test data
    batch_size = 32
    number_of_workers = 4
    data_loader_manager = DataLoaderManager(batch_size, valid_size=0, number_of_workers=number_of_workers)
    test_loader = data_loader_manager.get_test_loader()

    # Initialize the network and load saved model
    # model = Net()
    # model.load_state_dict(torch.load('models/originNet_2023-12-07_15-54-16.pt'))
    model = models.resnet18(pretrained=True)
    model.load_state_dict(torch.load('models/ResNet18_2023-12-07_20-51-09.pt'))

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model.cuda()

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    model.eval()

    # Calculate Test Loss and other metrics
    test_loss = compute_loss(model, test_loader, criterion, 'cuda' if train_on_gpu else 'cpu')
    test_accuracy = compute_accuracy(model, test_loader, 'cuda' if train_on_gpu else 'cpu')
    test_precision, test_recall, test_f1 = compute_precision_recall_fscore(model, test_loader,
                                                                           'cuda' if train_on_gpu else 'cpu')

    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

    predictions, actuals = make_predictions(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    predicted_classes, actual_classes = map_class_names(predictions, actuals)
    stats = class_statistics(predicted_classes, actual_classes)

    metrics.update(stats)

    overall_accuracy = calculate_accuracy(predicted_classes, actual_classes)
    metrics['overall_accuracy'] = overall_accuracy

    logger.log_metrics(0, metrics)
    recorder.update_metrics(0, metrics)

    recorder.save()


if __name__ == "__main__":
    evaluate()
