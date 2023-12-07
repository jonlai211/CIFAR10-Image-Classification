import torch.nn as nn
import torch.optim as optim


def get_loss_function(loss_function_name):
    if loss_function_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_function_name == 'sparse categorical cross-entropy':
        return nn.NLLLoss()  # make sure the output of the model is log_softmax
    elif loss_function_name == 'label_smoothing':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function_name}")


def get_optimizer(optimizer_name, model, learning_rate, momentum):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
