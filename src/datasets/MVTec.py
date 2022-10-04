import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.io import imread
from skimage.transform import resize


"""
MVTec datasets
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
        # print(self.img_files[idx])    #####
        img = Image.open(self.img_files[idx])
        # print(img.size, img.mode)    #####
        if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
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
        if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img_name = self.img_files[idx]

        # mask
        # h, w, _ = img.shape
        if img_name.split('/')[-2] == "good":
            mask = np.zeros(self.img_size)
        else:
            if "wine" in img_name:
                mask_path = img_name.replace("test", "ground_truth").split(".")[-2] + ".png"
            else:
                mask_path = img_name.replace("test", "ground_truth").split(".")[-2] + "_mask.png"
            mask = imread(mask_path, as_gray=True)
            # 0: Nearest-neighbor, ref: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
            mask = resize(mask, self.img_size, order=0)     # mask {0, 1}
        return img, mask, img_name

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
            print('loading image files ' + root)
            for file in files:
                if os.path.splitext(file)[-1] in ext and 'checkpoint' not in file:
                    images.append(os.path.join(root, file))
        return sorted(images)


if __name__ == "__main__":
    data_name = "leather"
    train_data_path = "/home/jie/Datasets/MVAomaly/"+ data_name + "/train"
    test_data_path = "/home/jie/Datasets/MVAomaly/" + data_name + "/test"
    train_data = NormalDataset(path=train_data_path, img_size=(256, 256), val=0.1, is_train=True, normalize=True)
    test_data = TestDataset(path=test_data_path, normalize=True)
    # print(train_data.img_files)
    # for img_file in test_data.img_files:
    #     print(img_file)

    # data loader
    from torch.utils.data import DataLoader
    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    for normal_img in train_data_loader:
        print("#############")
        print(normal_img.shape)


    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    for abnormal_img, mask, abnormal_img_name in test_data_loader:
        print("#############")
        print(abnormal_img_name[0])
        print(abnormal_img.shape)
        print(mask.shape)
        print(mask.max(), mask.min())





