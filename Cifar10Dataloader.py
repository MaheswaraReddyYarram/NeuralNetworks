import os
from PIL import Image


class Cifar10Dataloader():
    def __init__(self, images_dir, labels_csv, transform = None):
        self.images_dir = images_dir
        self.labels_csv = labels_csv
        self.transforms = transform

    def __len__(self):
        return len(self.labels_csv)

    #function to load each image since we don't have the default structure that PyTorch wants it's Datasets to have
    def __getitem__(self, idx):
        img_name = str(self.labels_csv.iloc[idx, 0]) + '.png'
        label = self.labels_csv.loc[idx, "label"]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)
        return img, label