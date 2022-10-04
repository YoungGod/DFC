import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from skimage.io import imread
from skimage.transform import resize


"""
Wine datasets
"""
class NormalDataset(Dataset):

    def __init__(self, path, img_size=(256, 256), val=0., is_train=True, normalize=True):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.img_files = self._get_image_files(path)
        self.len = len(self.img_files)
        self.img_size = img_size

        if val > 0:
            np.random.seed(0)
            self.img_files = np.random.permutation(self.img_files)
            num_val = int(self.len * val)
            if is_train:
                self.img_files = self.img_files[num_val:]
                self.len = self.len - num_val
            else:
                self.img_files = self.img_files[:num_val]
                self.len = num_val

        # transformer
        resize = transforms.Resize(size=img_size, interpolation=Image.NEAREST)
        if normalize:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([resize, transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        img = Image.open(self.img_files[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
            if "ipynb_checkpoints" in root:
                continue
            print('loading image files ' + root)
            for file in files:
                if os.path.splitext(file)[1] in ext:  # and "good" not in root
                    images.append(os.path.join(root, file))
        return sorted(images)


class TestDataset(Dataset):

    def __init__(self, path, img_size=(256, 256), normalize=True):
        self.img_files = self._get_image_files(path)
        self.len = len(self.img_files)
        self.img_size = img_size

        # transformer
        resize = transforms.Resize(size=img_size, interpolation=Image.NEAREST)
        if normalize:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([resize, transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        img = Image.open(self.img_files[idx])
        if self.transform is not None:
            img = self.transform(img)
        img_name = self.img_files[idx]

        # mask
        # h, w, _ = img.shape
        if img_name.split('/')[-2] == "good":
            mask = np.zeros(self.img_size)
        else:
            mask_path = img_name.replace("test", "test_gt").split(".")[-2] + ".png"
            mask = imread(mask_path, as_gray=True)
            # 0: Nearest-neighbor, ref: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
            mask = resize(mask, self.img_size, order=0)     # mask {0, 1}
        return img, mask, img_name

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
            if "ipynb_checkpoints" in root:
                continue
            print('loading image files ' + root)
            for file in files:
                if os.path.splitext(file)[1] in ext:
                    images.append(os.path.join(root, file))
        return sorted(images)


if __name__ == "__main__":
    data_name = "wine"
    train_data_path = "/home/jie/Datasets/wine_anomaly_cropped/train"
    test_data_path = "/home/jie/Datasets/wine_anomaly_cropped/test"
    train_data = NormalDataset(path=train_data_path, is_train=True, val=0.3)
    val_data = NormalDataset(path=train_data_path, is_train=False, val=0.3)
    test_data = TestDataset(path=test_data_path, normalize=False)
    print(train_data.img_files)
    print(val_data.img_files)
    print(set(train_data.img_files) & set(val_data.img_files))
    print(len(test_data.img_files))

    import matplotlib.pylab as plt
    # img = Image.open(train_data.img_files[0])
    # plt.imshow(img)
    # plt.show()

    # data loader
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    # train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    # for normal_img in tqdm(train_data_loader):
    #     print("#####Train########")
    #     print(normal_img.shape)

    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)
    i = 0
    for abnormal_img, mask, abnormal_img_name in tqdm(test_data_loader):
        i += 1
        print("#####Test########")
        print(abnormal_img_name[0])
        print(abnormal_img.shape)
        print(mask.shape)

        # if i % 50 == 0:
        #     img = abnormal_img.squeeze().permute(1,2,0).numpy()
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.figure()
        #     plt.imshow(mask.squeeze())
        #     plt.show()



