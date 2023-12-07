import torch


def make_predictions(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.view(-1).tolist())
            actuals.extend(target.view(-1).tolist())

    return predictions, actuals


def map_class_names(predictions, actuals):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_classes = [class_names[p] for p in predictions]
    actual_classes = [class_names[a] for a in actuals]
    return predicted_classes, actual_classes


def class_statistics(predicted_classes, actual_classes):
    stats = {class_name: {'total': 0, 'correct': 0, 'incorrect': 0} for class_name in set(actual_classes)}

    for pred, actual in zip(predicted_classes, actual_classes):
        stats[actual]['total'] += 1
        if pred == actual:
            stats[actual]['correct'] += 1
        else:
            stats[pred]['incorrect'] += 1

    for class_name, data in stats.items():
        data['accuracy'] = (data['correct'] / data['total']) if data['total'] > 0 else 0

    return stats


def calculate_accuracy(predicted_classes, actual_classes):
    correct_predictions = sum(p == a for p, a in zip(predicted_classes, actual_classes))
    total_predictions = len(actual_classes)
    return correct_predictions / total_predictions if total_predictions > 0 else 0
