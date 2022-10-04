#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets.MVTec import NormalDataset, TestDataset

from dfc import DFC
from utils import *


DEVICE = "cuda:0"
root_path = 'Project-DFC'

# data sets
textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
objects = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut','pill', 'screw', 'toothbrush', 'transistor', 'zipper', 'wine']
data_names = objects + textures


# params
TRAIN_BATCH_SIZE = 8
CNN_LAYERS = ("relu5_1", "relu5_2", "relu5_3", "relu5_4")
cnn_layers_textures = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
cnn_layers_objects = ("relu4_3", "relu4_4", "relu5_1", "relu5_2") 
IMG_SIZE = (256, 256)
LR = 1e-4

for data_name in data_names:
    
    # init model
    if data_name in textures:
        CNN_LAYERS = cnn_layers_textures
    else:
        CNN_LAYERS = cnn_layers_objects

    # init netwoks (where teacher is distilled from vgg19)
    model = DFC(device=DEVICE, cnn_layers=CNN_LAYERS, lr=LR, upsample="bilinear", anomaly_map_size=IMG_SIZE, 
                        root_path=root_path, data_name=data_name, model_name='DFC')
    # load model
    model.load_model()


    # data sets
    train_data_path = "/home/jie/Datasets/MVAnomaly/" + data_name + "/train/good"
    test_data_path = "/home/jie/Datasets/MVAnomaly/" + data_name + "/test"

    # training set for student (distill the one-class knowledge to student)
    data_train = NormalDataset(path=train_data_path, img_size=IMG_SIZE, val=0, is_train=True, normalize=True)
    train_loader = DataLoader(dataset=data_train, batch_size=8, shuffle=False, num_workers=1)

    # validation set
    data_val = NormalDataset(path=train_data_path, img_size=IMG_SIZE, val=0.05, is_train=False, normalize=True)
    val_loader = DataLoader(dataset=data_val, batch_size=8, shuffle=False, num_workers=1)

    # test set
    data_test = TestDataset(path=test_data_path, img_size=IMG_SIZE, normalize=True)
    test_loader = DataLoader(dataset=data_test, batch_size=1, shuffle=False)


    # Visualization
    # Anomaly Score
    for i, (img, mask, img_path) in enumerate(test_loader):
        img = img.to(DEVICE)
        # score
        score = model.compute_anomaly_score(img).data.cpu().numpy().squeeze()
        # gt mask
        mask = mask.squeeze().numpy()

        # visulize anomaly score
        img_path = img_path[0]
        anomaly_type = img_path.split('/')[-2]
        save_path = root_path + '/Results/{0}/{1}'.format(model.data_name, "_".join(model.feat_layers)) + '/score_map/' + anomaly_type

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        visualize_score(score, img_path, save_path, img_size=IMG_SIZE)


    # Ground Truth
    # visualize ground truth
    for i, (img, mask, img_path) in enumerate(test_loader):
        # gt mask
        mask = mask.squeeze().numpy()

        # visulize anomaly score
        img_path = img_path[0]
        if "good" in img_path:
            continue
        anomaly_type = img_path.split('/')[-2]
        save_path = root_path + '/Results/{0}/{1}'.format(model.data_name, "_".join(model.feat_layers)) + '/ground_truth/' + anomaly_type

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        visualize_gt(mask, img_path, save_path, img_size=IMG_SIZE)


    # Segmentation Map
    ious = []
    # visualize segmentation map
    for i, (img, mask, img_path) in enumerate(test_loader):

        img = img.to(DEVICE)
        # score
        score = model.compute_anomaly_score(img).data.cpu().numpy().squeeze()
        # gt mask
        mask = mask.squeeze().numpy()

        # visulize anomaly score
        img_path = img_path[0]
        if "good" in img_path:
            continue
        anomaly_type = img_path.split('/')[-2]
        save_path = root_path + '/Results/{0}/{1}'.format(model.data_name, "_".join(model.feat_layers)) + '/seg_map/' + anomaly_type

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        thred = 0.5
        score = normlize(score)
        pred_label = np.zeros_like(score)
        pred_label[score > thred] = 1
        pred_label[score <= thred] = 0
        visualize_segmap(pred_label, img_path, mask, save_path, thred=thred, img_size=IMG_SIZE)

        # per imag iou
        ious.append(iou(pred_label, mask))
    m_iou = np.array(ious).mean()
    print(data_name, ", Mean IOU:", m_iou)

    # save IOU
    save_path = root_path + '/metrics'
    if not os.path.exists(os.path.join(save_path, 'iou.csv')):
        with open(os.path.join(save_path, 'iou.csv'), 'w') as f:
            f.write('objects,iou\n')
    with open(os.path.join(save_path, 'iou.csv'), 'a+') as f:
        f.write(data_name + ',' +str(m_iou) +'\n')


