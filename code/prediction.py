import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import time

from cls_data import cls_data
from rn_cnn import rn_cnn


# *** the function of prediction *** #
def pre(model, test_data, device):
    correct = 0
    tolol = 0
    epoch_loss = []
    cor_total = 0
    cor_pred_right = 0
    broken_total = 0
    broken_pred_right = 0

    # model.eval()
    with torch.no_grad():
        for j, datas in enumerate(test_data):
            inputs, labels = datas
            inputs, labels = inputs.to(device), labels.to(device)

            torch.cuda.synchronize()
            start2 = time.time()
            pre = model(inputs)

            # *** Statistical the time for
            # one image of clusters of corn kernel *** #
            torch.cuda.synchronize()
            end2 = time.time()
            spend_time = end2 - start2
            print("the spent time of 65 image：{}s".format(spend_time))

            # *** Statistical predict Results *** #
            #         tolol += labels.size(0)
            #         correct += (y_pred == labels).sum().item()
            #         epoch_loss.append(loss.item())
            #
            #         this_batch_cor = labels.sum().item()
            #         cor_total += this_batch_cor
            #         broken_total += (len(labels) - this_batch_cor)
            #         cor_pred_right += (torch.nonzero((y_pred + labels) == 2)).shape[0]
            #         broken_pred_right += (torch.nonzero((y_pred + labels) == 0)).shape[0]
            #
            # return (100 * correct / tolol), tolol, correct, \
            #        (100*cor_pred_right/cor_total), cor_pred_right, cor_total, \
            #        (100*broken_pred_right/broken_total), broken_pred_right, broken_total,\
            #        np.mean(epoch_loss)

            # *** calculate breakage rate *** #
            y_pred = torch.argmax(pre, dim=1)
            cor_total += (torch.nonzero(y_pred == 1)).shape[0]
            broken_total += (torch.nonzero(y_pred == 0)).shape[0]
            for i, y in enumerate(y_pred):
                if y == 0:
                    img_arr = np.array(inputs[i].to("cpu"))
                    img_img = Image.fromarray(np.uint8((img_arr * 255).transpose(1, 2, 0))).convert('RGB')
                    img_img.save('./broken_img/{}.jpg'.format(64 * j + i))

        return broken_total / (cor_total + broken_total)


if __name__ == '__main__':

    # *** load model_weights *** #
    model_weights = "./save_weights/RN-CNN_best_epoch"

    # *** deal with dataset *** #
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224, 224)),
                                         ])
    test_dataset = cls_data("E:/CKCNN/Classification/test3", txt_name="test",
                            transforms=data_transform, test=True)
    test_data = DataLoader(test_dataset,
                           batch_size=65,
                           shuffle=False,
                           num_workers=1)

    # *** model *** #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = rn_cnn()
    model.load_state_dict(torch.load(model_weights))
    model.to(device)

    # all_ratio, all_nums, all_correct, \
    # good_ratio, good_right, good_all, \
    # broken_ratio, broken_right, broken_all, \
    # loss = pre(model,test_data, device)
    # print("测试集分类总精度{}，其中所有籽粒共{}，成功分类{}".format(all_ratio, all_nums, all_correct))
    # print("完整籽粒分类总精度{}，其中完整籽粒共{}，成功分类{}".format(good_ratio, good_all, good_right))
    # print("破损籽粒分类总精度{}，其中破损籽粒共{}玉米籽粒，成功分类{}".format(broken_ratio, broken_all, broken_right))

    # *** prediction *** #
    broken_rate = pre(model, test_data, device)
    print(broken_rate)
