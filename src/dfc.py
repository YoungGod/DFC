import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from tqdm import trange
from tqdm import tqdm

from vgg19 import VGG19
from vgg19_s import VGG19_S

from utils import *

from contextlib import contextmanager
@contextmanager
def task(_):
    yield

# backbone nets
#backbone_nets = {'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19}
backbone_nets = {'vgg19': VGG19}


class DFC():
    r"""
    Build muti-scale feature corresponding model based on VGG-feature maps
    """

    def __init__(self, backbone = 'vgg19', 
                 cnn_layers = ("relu1_2","relu2_2"), 
                 upsample = "bilinear",
                 anomaly_map_size = (256, 256), 
                 lr = 1e-4,
                 epochs = 100,
                 batch_size = 1,
                 loss_fn = nn.MSELoss(reduction='mean'),
                 optimizer = torch.optim.Adam,
                 data_name="",
                 root_path="",
                 random_feat_net=False, model_name='DFC', device='cpu'):
        self.device = torch.device(device)
        self.root_path = root_path
        self.data_name = data_name
        self.model_name = model_name

        print("Init feature net...")
        if random_feat_net:
            self.feature_extraction = backbone_nets[backbone](pretrain=False, gradient=False, pool='avg').to(self.device)   # random feature extraction net
        else:
            self.feature_extraction = backbone_nets[backbone](pretrain=True, gradient=False, pool='avg').to(self.device)   # pretrained feature extraction net
        
        print("Init matching net...")
        self.feature_matching = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)    # match net
        
        self.feat_layers = cnn_layers
        self.anomaly_map_size = anomaly_map_size
        self.upsample = upsample

        # Redefine training params
        self.loss = loss_fn
        self.lr = lr
        self.optimizer = optimizer(self.feature_matching.parameters(), lr=self.lr, weight_decay=1e-6)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, train_set, val_set=None, test_set=None):
        from torch.utils.data import DataLoader
        with task("datasets"):
            from datasets.augment_data import OESDataset
            # define OES dataset
            train_dataset = OESDataset(train_set, img_size=train_set.img_size, aug_data=None, normalize=True)
            train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)
            if val_set:
                val_dataset = OESDataset(val_set, img_size=val_set.img_size, aug_data=None, normalize=True)
                val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
            if test_set:
                test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

        with task("train"):
            # try to load model
            if self.load_model():
                print("Model loaded. Skip training.")
            # else: 
            # training
            self.feature_extraction.eval()
            self.feature_matching.train()
            print("Start training...")
            loss_avg = 0.
            loss_normal_avg = 0.
            loss_abnormal_avg = 0.

            for epoch in trange(self.epochs, leave=False):
                self.feature_matching.train()
                for normal, abnormal, mask in train_loader:
                    normal = normal.to(self.device)
                    abnormal = abnormal.to(self.device)
                    mask = mask.to(self.device)

                    self.optimizer.zero_grad()
                    with task("normal"):
                        surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                        pred = self.feature_matching(normal, self.feat_layers)
                        loss_normal = 0
                        for feat_layer in self.feat_layers:
                            loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                        loss_normal = loss_normal / len(self.feat_layers)

                    with task('abnormal'):
                        surrogate_label_abnormal = self.feature_extraction(abnormal, self.feat_layers)
                        pred = self.feature_matching(abnormal, self.feat_layers)
                        # abnormal feature inpainting or project to normal
                        loss_abnormal_1 = 0
                        for feat_layer in self.feat_layers:
                            loss_abnormal_1 += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                        loss_abnormal_1 = loss_abnormal_1 / len(self.feat_layers)

                    alpha = 1
                    loss = loss_normal + alpha*loss_abnormal_1
                    loss.backward()
                    self.optimizer.step()

                    # exponential moving average
                    loss_avg = loss_avg * 0.9 + float(loss.item()) * 0.1
                    loss_normal_avg = loss_normal_avg * 0.9 + float(loss_normal.item()) * 0.1
                    loss_abnormal = alpha*loss_abnormal_1 
                    loss_abnormal_avg = loss_abnormal_avg * 0.9 + float(loss_abnormal.item()) * 0.1
                print(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, loss_abnormal = {loss_abnormal_avg:.5f}")
                # tracking traing loss and val loss (for small experiments)
                if (epoch) % 2 == 0:
                    self.tracking_loss(epoch, f"{loss_avg:.5f},{loss_normal_avg:.5f},{loss_abnormal_avg:.5f}")
                    if val_loader:
                        self.tracking_val_loss(epoch, val_loader)
                # tracking metrics on test set & save model (for small experiments)
                if (epoch) % 10 == 0:
                    # save model
                    self.save_model()
                    if test_loader:
                        self.validation(epoch, test_loader)
            # save model
            self.save_model()
            print("Matching Net Trained.")

    def tracking_loss(self, epoch, loss):
        save_path = self.root_path + '/models/{0}/{1}'.format(self.data_name, "_".join(self.feat_layers))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out_file = os.path.join(save_path, '{}_epoch_train_loss.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",loss" + ",loss_normal" + ",loss_abnormal""\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(loss) + "\n")

    def tracking_val_loss(self, epoch, val_dataloader):
        self.feature_matching.eval()
        track_loss = []
        track_loss_normal = []
        track_loss_abnormal = []
        for normal, abnormal, mask in val_dataloader:
            normal = normal.to(self.device)
            abnormal = abnormal.to(self.device)
            mask = mask.to(self.device)
            with torch.no_grad():
                with task("normal"):
                    surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                    pred = self.feature_matching(normal, self.feat_layers)
                    loss_normal = 0
                    for feat_layer in self.feat_layers:
                        loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                    loss_normal = loss_normal / len(self.feat_layers)

                with task('abnormal'):
                    surrogate_label_abnormal = self.feature_extraction(abnormal, self.feat_layers)
                    pred = self.feature_matching(abnormal, self.feat_layers)
                    # abnormal feature inpainting or project to normal
                    loss_abnormal_1 = 0
                    for feat_layer in self.feat_layers:
                        loss_abnormal_1 += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                    loss_abnormal_1 = loss_abnormal_1 / len(self.feat_layers)


                loss = loss_normal + loss_abnormal_1
            track_loss.append(loss.item())
            track_loss_normal.append(loss_normal.item())
            track_loss_abnormal.append((loss_abnormal_1).item())
        loss_mean = np.array(track_loss).mean()
        loss_normal_mean = np.array(track_loss_normal).mean()
        loss_abnormal_mean = np.array(track_loss_abnormal).mean()

        # save the val loss
        save_path = self.root_path + '/models/{0}/{1}'.format(self.data_name, "_".join(self.feat_layers))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out_file = os.path.join(save_path, '{}_epoch_val_loss.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",loss" + ",loss_normal"+ ",loss_abnormal" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + f"{loss_mean:.5f},{loss_normal_mean:.5f},{loss_abnormal_mean:.5f}" + "\n")

    def save_model(self):
        save_path = self.root_path + '/models/{0}/{1}'.format(self.data_name, "_".join(self.feat_layers))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.feature_matching.state_dict(), save_path + '/match.pt')

    def load_model(self):
        student_path = self.root_path + '/models/{0}/{1}/match.pt'.format(self.data_name, "_".join(self.feat_layers))
        if os.path.exists(student_path):
            self.feature_matching.load_state_dict(torch.load(student_path))
            print("Model/Student {} loaded".format(student_path))
            return True
        else:
            return False

    def compute_anomaly_score(self, img):
        self.feature_extraction.eval()
        self.feature_matching.eval()

        img = img.to(self.device)
        with torch.no_grad():
            surrogate_label = self.feature_extraction(img, self.feat_layers)
            pred = self.feature_matching(img, self.feat_layers)
        anomaly_map = 0
        for feat_layer in self.feat_layers:
            # print(feat_layer)
            anomaly_map += F.interpolate(torch.pow(surrogate_label[feat_layer]-pred[feat_layer], 2).mean(1, keepdim=True), 
                                      size=self.anomaly_map_size, mode=self.upsample, align_corners=True)
        return anomaly_map

    def validation(self, epoch, test_dataloader):
        print("Validation AUC metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(test_dataloader):  # batch size is 1.
            i += 1
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # score
            score = self.compute_anomaly_score(img).data.cpu().numpy().squeeze()

            masks.append(mask)
            scores.append(score)
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)    # mask vale {0, 1}
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        scores = np.array(scores)

        # auc score
        auc_score, roc = auc_roc(masks, scores)
        # metrics over all data
        print("Epoch {0},  AUC: {1}".format(epoch, auc_score))

        # save
        save_path = self.root_path + '/models/{0}/{1}'.format(self.data_name, "_".join(self.feat_layers))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out_file = os.path.join(save_path, '{}_epoch_auc.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",AUC" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(auc_score) + "\n")
    
    def metrics_evaluation(self, test_dataloader, expect_fpr=0.3, pro_scale=True, max_step=200):
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score, average_precision_score
        from skimage import measure
        import pandas as pd
        print("Calculating AUC, IOU, PRO metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(test_dataloader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # anomaly score
            anomaly_map = self.compute_anomaly_score(img).data.cpu().numpy().squeeze()

            masks.append(mask)
            scores.append(anomaly_map)
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        scores = np.array(scores)
        
        # binary masks
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        
        # auc score (image level) for detection
        labels = masks.any(axis=1).any(axis=1)
        # preds = scores.mean(1).mean(1)
        preds = scores.max(1).max(1)    # for detection max score or mean score?
        det_auc_score = roc_auc_score(labels, preds)
        det_pr_score = average_precision_score(labels, preds)
        
        # auc score (per pixel level) for segmentation
        seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
        seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
        # metrics over all data
        print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
        print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")
        
        # per region overlap and per image iou
        max_th = scores.max()
        min_th = scores.min()
        delta = (max_th - min_th) / max_step
        
        ious_mean = []
        ious_std = []
        pros_mean = []
        pros_std = []
        threds = []
        fprs = []
        binary_score_maps = np.zeros_like(scores, dtype=np.bool)
        for step in range(max_step):
            thred = max_th - step * delta
            # segmentation
            binary_score_maps[scores <= thred] = 0
            binary_score_maps[scores > thred] = 1

            pro = []    # per region overlap
            iou = []    # per image iou
            # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
            # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
            for i in range(len(binary_score_maps)):    # for i th image
                # pro (per region level)
                label_map = measure.label(masks[i], connectivity=2)
                props = measure.regionprops(label_map)
                for prop in props:
                    x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]    # bug
                    cropped_mask = prop.filled_image    # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                    pro.append(intersection / prop.area)
                # iou (per image level)
                intersection = np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                union = np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                if masks[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                    iou.append(intersection / union)
            # against steps and average metrics on the testing data
            ious_mean.append(np.array(iou).mean())
            # print("per image mean iou:", np.array(iou).mean())
            ious_std.append(np.array(iou).std())
            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
            # fpr for pro-auc
            masks_neg = ~masks
            fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
            fprs.append(fpr)
            threds.append(thred)
            
        # as array
        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        pros_std = np.array(pros_std)
        fprs = np.array(fprs)
        
        ious_mean = np.array(ious_mean)
        ious_std = np.array(ious_std)
        
        # save results
        data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std',
                                                        'ious_mean', 'ious_std'])
        # save results
        save_path = self.root_path + '/Results/{0}/{1}'.format(self.data_name, "_".join(self.feat_layers))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_metrics.to_csv(os.path.join(save_path, 'thred_fpr_pro_iou.csv'), sep=',', index=False)

        
        # best per image iou
        best_miou = ious_mean.max()
        print(f"Best IOU: {best_miou:.4f}")
        
        # default 30% fpr vs pro, pro_auc
        idx = fprs <= expect_fpr    # find the indexs of fprs that is less than expect_fpr (default 0.3)
        fprs_selected = fprs[idx]
        fprs_selected = rescale(fprs_selected)    # rescale fpr [0,0.3] -> [0, 1]
        if pro_scale:
            pros_mean_selected = rescale(pros_mean[idx])    # need scale
        else:
            pros_mean_selected = pros_mean[idx]    # no scale  (correct?)
        pro_auc_score = auc(fprs_selected, pros_mean_selected)
        print("pro auc ({}% FPR):".format(int(expect_fpr*100)), pro_auc_score)

        # save results
        data = np.vstack([threds[idx], fprs[idx], pros_mean[idx], pros_std[idx]])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std'])
        df_metrics.to_csv(os.path.join(save_path, 'thred_fpr_pro_{}.csv'.format(expect_fpr)), sep=',', index=False)

        # save auc, pro as 30 fpr
        with open(os.path.join(save_path, 'pr_auc_pro_iou_{}.txt'.format(expect_fpr)), mode='a+') as f:
                f.write("det_pr, det_auc, seg_pr, seg_auc, seg_pro, seg_iou\n")
                f.write(f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_pr_score:.5f},{seg_auc_score:.5f},{pro_auc_score:.5f},{best_miou:.5f}")
        return f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_pr_score:.5f},{seg_auc_score:.5f},{pro_auc_score:.5f},{best_miou:.5f}"


if __name__ == "__main__":
    import time
    import numpy as np

    feature_layers=("relu1_1", "relu1_2", "relu2_1", "relu2_2", 
                    "relu3_1", "relu3_2", "relu3_3", "relu3_4",
                    "relu4_1", "relu4_2", "relu4_3", "relu4_4",
                    "relu5_1", "relu4_2", "relu5_3", "relu5_4")

    vgg19_layers = ("relu4_3", "relu4_4",
                    "relu5_1", "relu4_2")
    device = "cuda"

    print("Start..")
    
    model = DFC(cnn_layers=vgg19_layers, device=device)
    x = torch.randn(size=(1,3,256,256), dtype=torch.float).to(device)

    times = []
    for i in range(100):
        print(i)
        time_s = time.time()
        y = model.compute_anomaly_score(x)
        times.append(time.time() - time_s)
    print("cost:", np.mean(times[1:]))


