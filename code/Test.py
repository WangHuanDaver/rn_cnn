import numpy as np
import torch


# *** model validation *** #
def tests(model, criterion, test_data, device):
    correct = 0
    total = 0
    epoch_loss = []
    with torch.no_grad():
        for datas in test_data:
            inputs, labels = datas
            inputs, labels = inputs.to(device), labels.to(device)
            # label_copy = labels.clone()

            # ones = torch.sparse.torch.eye(2).to(device)
            # labels = ones.index_select(0, labels)
            pre = model(inputs)
            # loss_label = torch.max(pre, dim=1)[0]
            # labels = labels.float()
            # y_pred = y_pred.float()
            loss = criterion(pre, labels)

            y_pred = torch.argmax(pre, dim=1)
            total += labels.size(0)
            correct += (y_pred == labels).sum().item()
            epoch_loss.append(loss.item())
    print("Accuracy on test:{},  Loss on test:{}".format((100 * correct / tolol), np.mean(epoch_loss)))
    return (100 * correct / total), np.mean(epoch_loss)
