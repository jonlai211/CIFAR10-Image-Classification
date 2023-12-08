import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

base_path = os.path.join(os.path.dirname(__file__), '../../images/data_source')

originnet_train_path = os.path.join(base_path, 'train_originNet_bs64_ep300_2023-12-08_03-50-35.csv')
modifiednet_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_03-21-25.csv')
originnet_evaluate_path = os.path.join(base_path, 'evaluate_originNet_bs64_ep300_2023-12-08_04-16-47.csv')
modifiednet_evaluate_path = os.path.join(base_path, 'evaluate_modifiedNet_bs64_ep300_2023-12-08_03-49-49.csv')

md_lr1_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_03-21-25.csv')  # lr = 0.0001
md_lr2_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_12-09-52.csv')  # lr = 0.0005
md_lr3_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_09-46-25.csv')  # lr = 0.001
md_lr4_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_11-36-15.csv')  # lr = 0.005
md_lr5_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_11-00-07.csv')  # lr = 0.01

md_sgd_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_09-46-25.csv')  # sgd
md_adam_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_12-53-12.csv')  # adam
md_rmsprop_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_13-58-16.csv')  # rmsprop
md_adagrad_train_path = os.path.join(base_path, 'train_modifiedNet_bs64_ep300_2023-12-08_14-38-32.csv')  # adagrad


def plot_loss_comparison(origin_net_path, modified_net_path):
    origin_net_df = pd.read_csv(origin_net_path)
    modified_net_df = pd.read_csv(modified_net_path)

    origin_net_df['Model'] = 'OriginNet'
    modified_net_df['Model'] = 'ModifiedNet'

    combined_df = pd.concat([origin_net_df, modified_net_df])

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    sns.lineplot(x='epoch', y='train_loss', hue='Model', data=combined_df, ax=axes[0])
    axes[0].set_title('Training Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')

    sns.lineplot(x='epoch', y='valid_loss', hue='Model', data=combined_df, ax=axes[1])
    axes[1].set_title('Validation Loss Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Valid Loss')

    plt.tight_layout()
    plt.savefig('../../images/loss_comparison.png', format='png')
    # plt.savefig('loss_comparison.pdf', format='pdf')
    plt.show()


# plot_loss_comparison(originnet_train_path, modifiednet_train_path)

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def build_confusion_matrix(df, class_labels):
    confusion_matrix = pd.DataFrame(0, index=class_labels, columns=class_labels)
    for label in class_labels:
        class_data = ast.literal_eval(df[label].values[0])
        correct = class_data['correct']
        incorrect = class_data['incorrect']
        confusion_matrix.at[label, label] = correct
        for other_label in class_labels:
            if other_label != label:
                confusion_matrix.at[other_label, label] = incorrect / (len(class_labels) - 1)
    return confusion_matrix


def plot_confusion_matrices(originnet_evaluate_path, modifiednet_evaluate_path):
    originnet_df = pd.read_csv(originnet_evaluate_path)
    modifiednet_df = pd.read_csv(modifiednet_evaluate_path)

    originnet_matrix = build_confusion_matrix(originnet_df, class_labels)
    modifiednet_matrix = build_confusion_matrix(modifiednet_df, class_labels)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    sns.heatmap(originnet_matrix, annot=True, cmap='Blues', fmt='g', ax=axes[0])
    axes[0].set_title('OriginNet Confusion Matrix')
    sns.heatmap(modifiednet_matrix, annot=True, cmap='Blues', fmt='g', ax=axes[1])
    axes[1].set_title('ModifiedNet Confusion Matrix')

    for ax in axes:
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('../../images/confusion_matrices_comparison.png', format='png')
    plt.show()


# plot_confusion_matrices(originnet_evaluate_path, modifiednet_evaluate_path)

def plot_performance_comparison(originnet_evaluate_path, modifiednet_evaluate_path):
    originnet_df = pd.read_csv(originnet_evaluate_path)
    modifiednet_df = pd.read_csv(modifiednet_evaluate_path)

    metrics = ['test_precision', 'test_recall', 'test_f1']
    originnet_metrics = originnet_df.loc[0, metrics]
    modifiednet_metrics = modifiednet_df.loc[0, metrics]

    comparison_df = pd.DataFrame({
        'Metric': metrics,
        'OriginNet': originnet_metrics.values,
        'ModifiedNet': modifiednet_metrics.values
    })

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    comparison_df = comparison_df.melt(id_vars='Metric', var_name='Model', value_name='Score')
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=comparison_df)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.savefig('../../images/model_arch_performance_comparison.png', format='png')
    plt.show()


# plot_performance_comparison(originnet_evaluate_path, modifiednet_evaluate_path)


learning_rates = ['0.0001', '0.0005', '0.001', '0.005', '0.01']


def plot_loss_comparison_for_lr(*paths):
    dfs = []
    for lr, path in zip(learning_rates, paths):
        df = pd.read_csv(path)
        df['Learning Rate'] = lr
        dfs.append(df)

    combined_df = pd.concat(dfs)

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    sns.lineplot(x='epoch', y='train_loss', hue='Learning Rate', data=combined_df, ax=axes[0])
    axes[0].set_title('Training Loss Comparison for Different Learning Rates')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')

    sns.lineplot(x='epoch', y='valid_loss', hue='Learning Rate', data=combined_df, ax=axes[1])
    axes[1].set_title('Validation Loss Comparison for Different Learning Rates')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Valid Loss')

    plt.tight_layout()
    # Replace the path below with the actual path where you want to save the image
    plt.savefig('../../images/loss_comparison_lr.png', format='png')
    # plt.savefig('loss_comparison_lr.pdf', format='pdf')
    plt.show()


# plot_loss_comparison_for_lr(md_lr1_train_path, md_lr2_train_path, md_lr3_train_path, md_lr4_train_path,
#                             md_lr5_train_path)

optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad']


def plot_loss_comparison_for_optimizer(*paths):
    dfs = []
    for optimizer, path in zip(optimizers, paths):
        df = pd.read_csv(path)
        df['Optimizer'] = optimizer
        dfs.append(df)

    combined_df = pd.concat(dfs)

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    sns.lineplot(x='epoch', y='train_loss', hue='Optimizer', data=combined_df, ax=axes[0])
    axes[0].set_title('Training Loss Comparison for Different Optimizers')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')

    sns.lineplot(x='epoch', y='valid_loss', hue='Optimizer', data=combined_df, ax=axes[1])
    axes[1].set_title('Validation Loss Comparison for Different Optimizers')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Valid Loss')

    plt.tight_layout()
    # Replace the path below with the actual path where you want to save the image
    plt.savefig('../../images/loss_comparison_optimizer.png', format='png')
    # plt.savefig('loss_comparison_optimizer.pdf', format='pdf')
    plt.show()


plot_loss_comparison_for_optimizer(md_sgd_train_path, md_adam_train_path, md_rmsprop_train_path, md_adagrad_train_path)
