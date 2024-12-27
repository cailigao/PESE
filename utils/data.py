import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class CIL_Forgery(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    class_order = np.arange(12).tolist()

    def download_data(self):
        train_dir = "./data/CIL_Forgery/train/"
        test_dir = "./data/CIL_Forgery/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)