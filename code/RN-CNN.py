import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from Train import train
from Test import tests
from rn_cnn import rn_cnn
from cls_data import cls_data

if __name__ == '__main__':

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    # #*** deal with dataset ***# #
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.RandomVerticalFlip(0.5),
                                     transforms.Resize((224, 224)),
                                     ]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((224, 224)),
                                   ])
    }

    train_dataset = cls_data("./Per_Object",
                                      txt_name="train",
                                      transforms=data_transform["train"], test=None)
    test_dataset = cls_data("./Per_Object", txt_name="test",
                                     transforms=data_transform["val"], test=None)
    train_data = DataLoader(train_dataset,
                            batch_size=200,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset,
                           batch_size=64,
                           shuffle=False,
                           num_workers=4)

    # *** model,loss,opt,and so on *** #
    model = rn_cnn()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    temp = 10
    epochs = 20

    # *** model train and validation *** #
    for epoch in range(epochs):
        epoch_train_acc, epoch_train_loss = train(model, criterion, optimizer, epoch, train_data,
                                                  device)
        epoch_test_acc, epoch_test_loss = tests(model, criterion, test_data, device)

        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)
        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)

        if temp == 10:
            temp = epoch_test_loss
        if epoch_test_loss < temp:
            temp = epoch_test_loss
            torch.save(model.state_dict(), "./save_weights/RN-CNN_best_epoch")
    torch.save(model.state_dict(), "./save_weights/RN-CNN_last_epoch")


    plt.figure(0)
    plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
    plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
    plt.legend()
    plt.savefig("./Fig/RN-CNN_loss.png")

    plt.figure(1)
    plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
    plt.plot(range(1, epochs + 1), test_acc, label='test_acc')
    plt.legend()
    plt.savefig("./Fig/RN-CNN_acc.png")
