import numpy as np
import torch


# *** model train *** #
def train(model, criterion, optimizer, epoch, train_data, device):
    running_loss = []
    totol_nums = 0
    correct_nums = 0
    acc = []
    for i, (inputs, labels) in enumerate(train_data, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        label_pre = model(inputs)
        loss = criterion(label_pre, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = torch.argmax(label_pre, dim=1)
        totol_nums += labels.size(0)
        correct_nums += (y_pred == labels).sum().item()
        running_loss.append(loss.item())

    print("epoch{}，train acc:{}, train loss:{}".format(epoch, (100 * (correct_nums / totol_nums)), np.mean(running_loss)))
    return (100 * (correct_nums / totol_nums)), np.mean(running_loss)
