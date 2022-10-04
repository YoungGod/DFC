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

PRO_SCALE = False

for data_name in data_names:
    if data_name in textures:
        CNN_LAYERS = cnn_layers_textures
    else:
        CNN_LAYERS = cnn_layers_objects
    # data
    test_data_path = "/home/jie/Datasets/MVAnomaly/" + data_name + "/test"

    # test set
    data_test = TestDataset(path=test_data_path, img_size=IMG_SIZE, normalize=True)
    test_loader = DataLoader(dataset=data_test, batch_size=1, shuffle=False)

    # init netwoks (where teacher is distilled from resnet18)
    model = DFC(device=DEVICE, cnn_layers=CNN_LAYERS, lr=LR, upsample="bilinear", anomaly_map_size=IMG_SIZE, 
                        root_path=root_path, data_name=data_name, model_name='DFC')
    model.load_model()
    
    # testing
    import time
    time_s = time.time()
    metrics = model.metrics_evaluation(test_dataloader=test_loader, expect_fpr=0.3, pro_scale=PRO_SCALE, max_step=5000)
    print("Cost total time {}s".format(time.time() - time_s))

    # saving all the metrics
    save_path = os.path.join(root_path, "metrics")
    file_name = "metrics.csv"
    if not os.path.exists(os.path.join(save_path, file_name)):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, file_name), mode='w') as f:
            f.write("catetory,det_pr,det_auc,seg_pr,seg_auc,seg_pro,seg_iou\n")
    with open(os.path.join(save_path, file_name), mode='a+') as f:
        f.write(data_name + "," + metrics + "\n")
