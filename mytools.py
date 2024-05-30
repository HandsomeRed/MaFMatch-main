from medpy import metric
import torch
import cv2
import numpy as np


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=[1, 10, 1, 1])
        asd = metric.binary.asd(pred, gt, voxelspacing=[1, 10, 1, 1])
        return dice, hd95, asd
    else:
        return 0, 50, 10


# def calculate_metric_percase(pred, gt):
#     # pred[pred > 0] = 1
#     # gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         asd = metric.binary.asd(pred, gt)
#         return dice, hd95, asd
#     else:
#         return 0, 50, 10


def calculate_metric_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0


def calculate_metric_hd95(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=[1, 10, 1, 1])
        return hd95
    else:
        return 50




def calculate_metric_asd(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        asd = metric.binary.asd(pred, gt, voxelspacing=[1, 10, 1, 1])
        return asd
    else:
        return 100


def calculate_metric_iou(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
