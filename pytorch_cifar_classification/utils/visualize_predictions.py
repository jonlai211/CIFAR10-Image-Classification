import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pytorch_cifar_classification.utils.predictor import make_predictions
from pytorch_cifar_classification.models.modified_net import Net
from pytorch_cifar_classification.datasets.data_loader_manager import DataLoaderManager
import seaborn as sns


def map_class_names(indices, class_labels):
    return [class_labels[i] for i in indices]


def visualize_and_save_predictions(model, test_loader, class_labels, num_images=5, save_path='../../images/'):
    sns.set(style='whitegrid', context='paper', palette='muted')

    predictions, actuals = make_predictions(model, test_loader)
    predicted_classes = map_class_names(predictions, class_labels)
    actual_classes = map_class_names(actuals, class_labels)

    correct_indices = [i for i, (p, a) in enumerate(zip(predicted_classes, actual_classes)) if p == a]
    incorrect_indices = [i for i, (p, a) in enumerate(zip(predicted_classes, actual_classes)) if p != a]

    # Save correct predictions
    fig, ax = plt.subplots(1, num_images, figsize=(10, 2), dpi=300)
    plt.suptitle("Correct Predictions", fontsize=12)
    for i in range(num_images):
        img = test_loader.dataset[correct_indices[i]][0].numpy().transpose((1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_xlabel(f'Pred: {predicted_classes[correct_indices[i]]}\nActual: {actual_classes[correct_indices[i]]}',
                         fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{save_path}correct_predictions.png', format='png')
    plt.close()

    # Save incorrect predictions
    fig, ax = plt.subplots(1, num_images, figsize=(10, 2), dpi=300)
    plt.suptitle("Incorrect Predictions", fontsize=12)
    for i in range(num_images):
        img = test_loader.dataset[incorrect_indices[i]][0].numpy().transpose((1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_xlabel(
            f'Pred: {predicted_classes[incorrect_indices[i]]}\nActual: {actual_classes[incorrect_indices[i]]}',
            fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{save_path}incorrect_predictions.png', format='png')
    plt.close()


model = Net()
model.load_state_dict(torch.load('../../models/modifiedNet_bs64_ep300_2023-12-08_10-19-07.pt'))
test_loader = DataLoaderManager(batch_size=32, valid_size=0, number_of_workers=4).get_test_loader()

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
visualize_and_save_predictions(model, test_loader, class_labels)
