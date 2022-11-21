from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


# *** dataset making *** #
class cls_data(Dataset):
    def __init__(self, root, txt_name: str = "train", transforms=None, test=None):
        if test == None:
            file_path = os.path.join("./rn-cnn-dataset", txt_name + "_idx.txt")
            with open(file_path, "r") as f:
                self.file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            self.img_path = [os.path.join(root, x + ".jpg") for x in self.file_names]

            img_target = pd.read_excel("./rn-cnn-dataset/new_target.xls", sheet_name="Sheet1")

            self.dic = dict(zip(img_target["img_nums"], img_target["target"]))

            self.transform = transforms
            self.test = test

        else:
            file_path = root
            self.file_names = os.listdir(file_path)
            self.img_path = [os.path.join(root, x) for x in self.file_names]

            img_target = pd.read_excel("E:/CKCNN/Classification/test_target2.xls", sheet_name="Sheet1")
            self.dic = dict(zip(img_target["img_nums"], img_target["target"]))

            self.transform = transforms
            self.test = test

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.test == None:
            target = self.dic[int(self.file_names[item])]
        else:
            a = self.file_names[item]
            target = self.dic[int(a[0: a.rfind('.')])]

        return img, target

    def __len__(self):
        return len(self.img_path)
