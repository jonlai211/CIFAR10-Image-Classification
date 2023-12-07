from pytorch_cifar_classification.models.origin_net import Net
from train import train
from evaluate import evaluate

batch_size = 10
n_epochs = 10
loss_function_name = 'cross_entropy'
optimizer_name = 'sgd'
learning_rate = 0.0001
momentum = 0.9

model_name = "originNet_bs"+str(batch_size)+"_ep"+str(n_epochs)
model = Net()

train(model_name, model, batch_size, n_epochs, loss_function_name, optimizer_name, learning_rate, momentum)
evaluate(model_name, model, batch_size)
