import numpy as np
import os
import cv2
from skimage.io import imread, imsave, imshow
from sklearn.metrics import roc_auc_score, roc_curve


def normalize01(x):
    """ Normalize x to [0, 1]
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def visulization(img_file, mask_path, score_map_path, saving_path):
    # image name
    img_name = img_file.split("/")
    img_name = "-".join(img_name[-2:])

    # image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    #     imsave("feature_maps/Results/gt_image/{}".format(img_name), image)

    # mask
    mask_file = os.path.join(mask_path, img_name)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # binary score {0, 255}
    score_file = os.path.join(score_map_path, img_name)
    score = cv2.imread(score_file, cv2.IMREAD_GRAYSCALE)
    img = img[:, :, ::-1]  # bgr to rgb
    img[..., 1] = np.where(score == 255, 255, img[..., 1])

    # save
    imsave(os.path.join(saving_path, "{}".format(img_name)), img)


def spec_sensi_acc_iou_auc(mask, binary_score, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    binary_score[binary_score > 0.5] = 1
    binary_score[binary_score <= 0.5] = 0

    gt_n = mask == 0
    pred_n = binary_score == 0
    gt_p = mask == 1
    pred_p = binary_score == 1

    specificity = np.sum(gt_n * pred_n) / np.sum(gt_n)
    sensitivity = np.sum(gt_p * pred_p) / np.sum(gt_p)
    accuracy = (np.sum(gt_p * pred_p) + np.sum(gt_n * pred_n)) / (np.sum(gt_p) + np.sum(gt_n))
    # coverage = np.sum(score * mask) / (np.sum(score) + np.sum(mask))

    intersection = np.logical_and(mask, binary_score)
    union = np.logical_or(mask, binary_score)
    iou_score = np.sum(intersection) / np.sum(union)

    auc_score = roc_auc_score(mask.ravel(), score.ravel())

    return specificity, sensitivity, accuracy, iou_score, auc_score


def spec_sensi_acc_riou_auc(mask, binary_score, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    binary_score[binary_score > 0.5] = 1
    binary_score[binary_score <= 0.5] = 0

    gt_n = mask == 0
    pred_n = binary_score == 0
    gt_p = mask == 1
    pred_p = binary_score == 1

    specificity = np.sum(gt_n * pred_n) / np.sum(gt_n)      # recall for negtive
    # specificity = np.sum(gt_p * pred_p) / np.sum(pred_p)    # precision
    sensitivity = np.sum(gt_p * pred_p) / np.sum(gt_p)      # recall for positive
    accuracy = (np.sum(gt_p * pred_p) + np.sum(gt_n * pred_n)) / (np.sum(gt_p) + np.sum(gt_n))
    # coverage = np.sum(score * mask) / (np.sum(score) + np.sum(mask))

    intersection = np.logical_and(mask, binary_score)
    union = np.logical_or(mask, binary_score)
    # iou_score = np.sum(intersection) / np.sum(union)
    iou_score = np.sum(intersection) / np.sum(mask)    # relative iou

    auc_score = roc_auc_score(mask.ravel(), score.ravel())

    fpr, tpr, thresholds = roc_curve(mask.ravel(), score.ravel(), pos_label=1)

    return specificity, sensitivity, accuracy, iou_score, auc_score, [fpr, tpr, thresholds]


def auc_roc(mask, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    auc_score = roc_auc_score(mask.ravel(), score.ravel())
    fpr, tpr, thresholds = roc_curve(mask.ravel(), score.ravel(), pos_label=1)

    return auc_score, [fpr, tpr, thresholds]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def normlize(x, r_min=0, r_max=1):
    min_x = x.min()
    max_x = x.max()
    return r_min + (x - min_x) / (max_x - min_x) * r_max

def iou(pred_label, mask):
    intersection = np.logical_and(pred_label, mask).astype(np.float32).sum()
    union = np.logical_or(pred_label, mask).astype(np.float32).sum()
    return intersection / union

# visulization function
def visualize_score(score, img_path, save_path, img_size=(256,256)):
    """
    Visuliaze anomaly score.
    """
    def normalize(x):
        return x/x.max()

    score = normalize(score)*255
    score = score.astype(np.uint8)
    heatmap = cv2.applyColorMap(score, cv2.COLORMAP_JET) 

    # img
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size[1], img_size[0]), cv2.INTER_NEAREST)
    superimposed_img = heatmap * 0.7 + img * 0.8

    # save
    img_name = img_path.split('/')[-1]
    cv2.imwrite(save_path + "/score_map_" + img_name, superimposed_img)

def visualize_gt(mask, img_path, save_path, img_size=(256,256)):
    """
    Visuliaze ground truth anomaly as red color.
    """
    def normalize(x):
        return x/x.max()
    # mask
    mask = normalize(mask)*255
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # img
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size[1], img_size[0]), cv2.INTER_NEAREST)

    cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=-1)

    # save
    img_name = img_path.split('/')[-1]
    cv2.imwrite(save_path + "/gt_" + img_name, img)

def visualize_segmap(pred_label, img_path, mask, save_path, thred=0.3, img_size=(256,256)):
    """
    Visualize segmentation map.
    """
    def normalize(x):
        return x/x.max()
    # image
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]), cv2.INTER_NEAREST)

    # mask
    mask = normalize(mask)*255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (img_size[1], img_size[0]), cv2.INTER_NEAREST)

    # gt contours as closed red lines 
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # predict label area filling with green
    img[..., 1] = np.where(pred_label == 1, 255, img[..., 1])    # set 255 in green channel

    # save
    img_name = img_path.split('/')[-1]
    cv2.imwrite(save_path + "/seg_map_thred_{}_".format(str(thred))+img_name, img)