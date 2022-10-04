import torch
import torch.utils.data
import numpy as np
import os
from scipy.io import loadmat
from PIL import Image
import random
import torchvision.transforms as transforms


class ILSVRC(torch.utils.data.Dataset):
    def __init__(self,ilsvrc_data_path, meta_path, transform=None, val=True):
        
        if val:
            synset_dir=os.path.join(ilsvrc_data_path,'val')
            print('loading val images ...')
        else:
            synset_dir=os.path.join(ilsvrc_data_path,'train')
            print('loading train images ...')
        self.samples = self.get_image_label(synset_dir, meta_path)    # tuples: (img_path, label)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
#         self.transform = transforms.Compose([transforms.RandomCrop(), transforms.ToTensor()])
        # self.transform = transform    # we define transforms outside the data set
       
    def __getitem__(self, index):
        sample = self.samples[index]
        # print(sample)
        # read and transform
        img = self.preprocess_image(sample[0])    # PIL image
        # print(img)
        # img=np.transpose(img,(2,0,1))
        # img = torch.from_numpy(img)
        img = np.array(img)
        img = self.transform(img)

        label = sample[1]
        return img, label   
        # return img    # only use the img

    def __len__(self):
        return len(self.samples)

    def preprocess_image(self,image_path):
        """ It reads an image, it resize it to have the lowest dimesnion of 256px,
            it randomly choose a 224x224 crop inside the resized image and normilize the numpy 
            array subtracting the ImageNet training set mean
            Args:
                images_path: path of the image
            Returns:
                cropped_im_array: the numpy array of the image normalized [width, height, channels]
        """
        # IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format
        # print(image_path)
        img = Image.open(image_path).convert('RGB')
        # print(img)
        # resize of the image (setting lowest dimension to 256px)
        if img.size[0] < img.size[1]:
            h = int(float(256 * img.size[1]) / img.size[0])
            img = img.resize((256, h), Image.ANTIALIAS)    # 抗锯齿
        else:
            w = int(float(256 * img.size[0]) / img.size[1])
            img = img.resize((w, 256), Image.ANTIALIAS)

        # in case when the image size < (256, 256)
        if img.size[0] < 256 or img.size[1] < 256:
            img = img.resize((256, 256), Image.ANTIALIAS)

        # random 244x224 patch
        x = random.randint(0, img.size[0] - 224)
        y = random.randint(0, img.size[1] - 224)
        img_cropped = img.crop((x, y, x + 224, y + 224))
        
        # data augmentation: flip left right
        if random.randint(0, 1) == 1:
            img_cropped = img_cropped.transpose(Image.FLIP_LEFT_RIGHT)

        # cropped_im_array = np.array(img_cropped, dtype=np.float32)
        #
        # for i in range(3):
        #     cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]
        #
        # return cropped_im_array/225
        return img_cropped

    def load_imagenet_meta(self, meta_path):
        """ It reads ImageNet metadata from ILSVRC 2012 dev tool file
            Args:
                meta_path: path to ImageNet metadata file
            Returns:
                wnids: list of ImageNet wnids labels (as strings)
                words: list of words (as strings) referring to wnids labels and describing the classes 
        """
        metadata = loadmat(meta_path, struct_as_record=False)

        # ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
        synsets = np.squeeze(metadata['synsets'])
        ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))[0:1000]
        wnids = np.squeeze(np.array([s.WNID for s in synsets]))[0:1000]
        # words = np.squeeze(np.array([s.words for s in synsets]))
        return dict(zip(wnids, ids))

    def get_image_label(self, data_path, meta_path):
        wnid_id = self.load_imagenet_meta(meta_path)    # dict
        img_file_label = []
        for wnid in wnid_id.keys():
            img_dir = os.path.join(data_path, str(wnid))
            img_names = os.listdir(img_dir)
            for img_name in img_names:
                img_file_dir = os.path.join(img_dir, img_name)
                label = wnid_id[wnid] - 1  # class label: 0~999
                img_file_label.append((img_file_dir, label))
        return img_file_label


if __name__=='__main__':

    data_path = "/home/jie/Datasets/ILSVRC2012"
    meta_path = "/home/jie/Datasets/ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.mat"
    train_dataset = ILSVRC(ilsvrc_data_path=data_path, meta_path=meta_path, val=False)
    val_dataset = ILSVRC(ilsvrc_data_path=data_path, meta_path=meta_path, val=True)
    from torch.utils.data import  DataLoader

    # print(train_dataset.samples[:100])
    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=1, shuffle=True,
                                           num_workers=0)

    img, label=train_dataset[100]
    print(img.shape,label)
    print(len(train_dataset))

    i = 0
    for img, label in train_loader:
        if i > 10:
            break
        i += 1
        print(img.shape, label)



    # metadata = loadmat(meta_path, struct_as_record=False)
    # print(sorted(metadata.keys()))
    # # print(metadata['synsets'])
    # print(len(metadata['synsets']))
    # synsets = metadata['synsets'].squeeze()
    # print("ID:", synsets[0].ILSVRC2012_ID)
    # print("WIND:", synsets[0].WNID)
    # # print("words:", metadata['synsets'][0][2])

    # metadata = loadmat(meta_path, struct_as_record=False)
    #
    # # ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
    # synsets = np.squeeze(metadata['synsets'])
    # ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))[0:1000]
    # wnids = np.squeeze(np.array([s.WNID for s in synsets]))[0:1000]
    # wnid_id = dict(zip(wnids, ids))
    # print(wnid_id)
    #
    # img_dirs = [os.path.join(data_path, str(wnid)) for wnid in wnids]
    # print(img_dirs[:10])
    #
    # i = 0
    # img_file_label = []
    # for wnid in wnid_id.keys():
    #     img_dir = os.path.join(data_path, str(wnid))
    #     img_names = os.listdir(img_dir)
    #     for img_name in img_names:
    #         if i>1:
    #             i += 1
    #         img_file_dir = os.path.join(img_dir, img_name)
    #         label = wnid_id[wnid] - 1    # class label: 0~999
    #         img_file_label.append((img_file_dir, label))
    # print(img_file_label[0:10])

"""
    synsets{
        ILSVRC2012_ID
        WNID
        words
        gloss
        num_children
        children
        wordnet_height
        num_train_images
    }
"""