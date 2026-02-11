import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os
import argparse
import cv2
from decoder import self_net
from dataset import NPY_datasets
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='D:\\python_rtobdt\\Lqcnet_Rail\\', help='test dataset path')
parser.add_argument('--testsize', type=int, default=352, help='testing size')

opt = parser.parse_args(args=[])

dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')

# load the model
model = self_net()
model.cuda()
checkpoint = ''

model.load_state_dict(torch.load(checkpoint))


def compute_iou(pred, target):
    """
    计算两个二维NumPy数组的IoU（交并比）。

    参数:
        pred (np.ndarray): 预测的二维数组，通常为二值掩码。
        target (np.ndarray): 目标的二维数组，通常为二值掩码。

    返回:
        float: IoU值。
    """
    # 计算交集
    intersection = np.sum(target * pred)

    # 计算并集
    union = np.sum(target) + np.sum(pred) - intersection

    # 数值稳定性处理：避免除零
    if union == 0:
        return 0.0

    # 计算IoU
    iou = intersection / union
    return iou


# def compute_pa(pred, target, num_classes=2):
#     intersection = np.logical_and(pred, target).sum()
#     union = target.sum()
#     pa = intersection / (union + 1e-8)
#     return pa

# test
# test_datasets = ['NJU2K','NLPR', 'SIP', 'STERE','DUT-RGBD','ReDWebTest',]


test_datasets = ['NEU RSDDS-AUG', ]
# test_datasets = ['NRSD-MN', ]
# test_datasets = ['Crack500', ]
# test_datasets = ['IPSS', ]

for dataset in test_datasets:
    #save_path = './test_maps_gai/'+dataset+"/"
    save_path = ''
    save_path_b = ''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_b):
        os.makedirs(save_path_b)

    data_path = dataset_path + dataset +"\\"

    val_dataset = NPY_datasets(data_path, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=0,
                            drop_last=False)
    print('val_loader:', len(val_loader))

    time_sum = 0
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        iou_sum = 0
        # pa = 0
        for data in tqdm(val_loader):
            img, msk, name = data
            img = img.cuda(non_blocking=True).float()

            time_start = time.time()

            out1, out2, out3, out4, out5, b1, b2, b3, b4, b5 = model(img, 200, [0, 18])

            time_end = time.time()
            time_sum = time_sum + (time_end - time_start)

            msk = np.asarray(msk.squeeze(), np.float32)
            res = out1
            res = F.interpolate(res, size=msk.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # print('save img to: ', save_path + name[0])
            cv2.imwrite(save_path + name[0], res * 255)

            # mIoU Calculation
            # binarized_res = (res > 0.5).astype(np.uint8)
            # binarized_msk = (msk > 0.5).astype(np.uint8)
            iou_sum += compute_iou(res, msk)
            # pa += compute_pa(binarized_res, binarized_msk)

            mae_sum += np.sum(np.abs(res - msk)) * 1.0 / (msk.shape[0] * msk.shape[1])

            b1 = F.interpolate(b1, size=msk.shape, mode='bilinear', align_corners=True)
            b1 = b1.data.cpu().numpy().squeeze()
            b1 = (b1 - b1.min()) / (b1.max() - b1.min() + 1e-8)
            cv2.imwrite(save_path_b + name[0], b1 * 255)

        print('Running time {:.5f}'.format(time_sum / len(val_loader)))
        print('Average speed: {:.4f} fps'.format(len(val_loader) / time_sum))
        mae = mae_sum / len(val_loader)
        miou = iou_sum / len(val_loader)
        # pa = pa / len(val_loader)
        log_info = f'dataset:{dataset} MAE:{mae:.4f} mIoU:{miou:.4f}'
        print(log_info)

print('Test Done!')