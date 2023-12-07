import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_cifar_classification.datasets.data_loader_manager import DataLoaderManager
from pytorch_cifar_classification.models.origin_net import Net
from pytorch_cifar_classification.utils.metrics import compute_accuracy, compute_loss, compute_precision_recall_fscore
from pytorch_cifar_classification.utils.logger import Logger
from pytorch_cifar_classification.utils.recorder import Recorder
from pytorch_cifar_classification.utils.saver import save_model
import torchvision.models as models
from pytorch_cifar_classification.utils.options import get_loss_function, get_optimizer


def train(model_name="originNet", model=Net(), batch_size=32, n_epochs=150, loss_function_name='cross_entropy',
          optimizer_name='sgd', learning_rate=0.0001, momentum=0.9):

    logger = Logger(model_name, mode='train')
    logger.log_message("Start training ...")
    recorder = Recorder(model_name, mode='train')

    # Initialize DataLoaderManager
    # batch_size = 32
    valid_size = 0.2
    number_of_workers = 4
    data_loader_manager = DataLoaderManager(batch_size, valid_size, number_of_workers)
    train_loader = data_loader_manager.get_train_loader()
    valid_loader = data_loader_manager.get_valid_loader()

    # n_epochs = 150

    # Initialize the network
    # model = Net()
    # model = models.resnet18(pretrained=True)
    # model = models.mobilenet_v2(pretrained=True)
    # model.classifier[1] = torch.nn.Linear(model.last_channel, 10)

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model.cuda()
    valid_loss_min = np.Inf

    # specify loss function
    criterion = get_loss_function(loss_function_name)

    # specify optimizer by Stochastic Gradient Descent
    optimizer = get_optimizer(optimizer_name, model, learning_rate, momentum)

    logger.log_message(
        "batch_size: {}\n"
        "valid_size: {}\n"
        "number_of_workers: {}\n"
        "n_epochs: {}\n"
        "train_on_gpu: {}\n"
        "optimizer: {}\n"
        "criterion: {}\n".format(
            batch_size, valid_size, number_of_workers,
            n_epochs, train_on_gpu, optimizer, criterion
        )
    )

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()

        # Calculate Train Loss and Validation Loss
        train_loss = compute_loss(model, train_loader, criterion, 'cuda' if train_on_gpu else 'cpu')
        valid_loss = compute_loss(model, valid_loader, criterion, 'cuda' if train_on_gpu else 'cpu')

        # Calculate Train Accuracy and Validation Accuracy
        train_accuracy = compute_accuracy(model, train_loader, 'cuda' if train_on_gpu else 'cpu')
        valid_accuracy = compute_accuracy(model, valid_loader, 'cuda' if train_on_gpu else 'cpu')

        # Calculate Precision, Recall and F1 Score
        train_precision, train_recall, train_f1 = compute_precision_recall_fscore(model, train_loader,
                                                                                  'cuda' if train_on_gpu else 'cpu')
        valid_precision, valid_recall, valid_f1 = compute_precision_recall_fscore(model, valid_loader,
                                                                                  'cuda' if train_on_gpu else 'cpu')

        metrics = {
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_accuracy': train_accuracy,
            'valid_accuracy': valid_accuracy,
            'train_precision': train_precision,
            'valid_precision': valid_precision,
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_f1': train_f1,
            'valid_f1': valid_f1
        }

        logger.log_metrics(epoch, metrics)
        recorder.update_metrics(epoch, metrics)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            logger.log_message(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))

        recorder.save()

    save_model(model, model_name)


# if __name__ == "__main__":
#     train()
