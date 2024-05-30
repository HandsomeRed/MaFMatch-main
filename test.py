import argparse
import logging
import os
import pprint
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.acdc import ACDCDataset
from model.unet import UNet
#from model.CBAM_unet import UNet
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.dist_helper import setup_distributed

from medpy import metric
import mytools

parser = argparse.ArgumentParser(description='MaFmatch')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet(in_chns=1, class_num=cfg['nclass'])
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 批量归一化
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])

    # 无标签训练集
    trainset_u = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    # 带标签训练集
    trainset_l = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    # 验证集
    valset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val')

    # 这些是分布式的内容
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True,
                                   sampler=trainsampler_u_mix)  # drop_last
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    print(args.save_path)
    # 断点续练
    if os.path.exists(os.path.join(args.save_path, 'best_MeanDice.pth')):

        checkpoint = torch.load(os.path.join(args.save_path, 'best_MeanDice.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

        model.eval()
        dice_class = [0] * 3
        dice_class1 = [0] * 3

        hd95_class = [0] * 3
        asd_class = [0] * 3
        iou_class = [0] * 3

        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

                img = img.permute(1, 0, 2, 3)  # 维度换位
                pred = model(img)  # torch.Size([10, 4, 256, 256])
                pred = F.interpolate(pred, (h, w), mode='bilinear',
                                     align_corners=False)  # 上采样 torch.Size([10, 4, 224, 222])
                pred = pred.argmax(dim=1).unsqueeze(0)  # torch.Size([1, 10, 224, 222])

                # 按类别预测
                for cls in range(1, cfg['nclass']):

                    inter = ((pred == cls) * (mask == cls)).sum().item()  # 7853
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls - 1] += 2.0 * inter / union

                    dice_class1[cls - 1] += mytools.calculate_metric_dice(pred == cls, mask == cls)

                    hd95_class[cls - 1] += mytools.calculate_metric_hd95(pred == cls, mask == cls)
                    asd_class[cls - 1] += mytools.calculate_metric_asd(pred == cls, mask == cls)
                    iou_class[cls - 1] += mytools.calculate_metric_iou(pred == cls, mask == cls)

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)

        dice_class1 = [dice * 100.0 / len(valloader) for dice in dice_class1]
        mean_dice1 = sum(dice_class1) / len(dice_class1)
        print("dice_class1:" + str(dice_class1))
        print("mean_dice1:" + str(mean_dice1))

        hd95_class = [hd95 / len(valloader) for hd95 in hd95_class]
        mean_hd95 = sum(hd95_class) / len(hd95_class)
        print("hd95_class:"+str(hd95_class))
        print("mean_hd95:" + str(mean_hd95))

        asd_class = [asd / len(valloader) for asd in asd_class]
        mean_asd = sum(asd_class) / len(asd_class)
        print("asd_class:"+str(asd_class))
        print("mean_asd:" + str(mean_asd))

        iou_class = [iou / len(valloader) for iou in iou_class]
        mean_iou = sum(iou_class) / len(iou_class)
        print("iou_class:" + str(iou_class))
        print("mean_iou:" + str(mean_iou))

        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: ''{:.2f}'.format(cls_idx,
                                                                                                CLASSES[cfg['dataset']][
                                                                                                    cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))






if __name__ == '__main__':
    main()
