import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix


# from utils import save_imgs


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer):
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()
        images, targets, points = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        points = points.cuda(non_blocking=True).float()

        out1, out2, out3, out4, out5, b1, b2, b3, b4, b5 = model(images, epoch, [0, 18])
        # loss = criterion(out1, targets) + 0.9 * criterion(b1_1, points) + 0.8 * criterion(b2_2, points) + 0.8 * criterion(b2_3, points)
        loss = criterion(out1, out2, out3, out4, out5, b1, b2, b3, b4, b5, targets, points)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss.item(), global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter: {iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    log_fin = f'train: epoch {epoch}, train_loss: {np.mean(loss_list):.4f}'
    logger.info(log_fin)
    writer.add_scalar('train_loss', np.mean(loss_list), global_step=epoch)

    return step


# def compute_iou(pred, target, num_classes=2):
#     intersection = np.logical_and(pred, target).sum()
#     union = np.logical_or(pred, target).sum()
#     iou = intersection / (union + 1e-8)
#     return iou


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

def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    with torch.no_grad():
        iou_sum = 0
        mae_sum = 0
        for data in tqdm(test_loader):
            img, msk, name = data
            img = img.cuda(non_blocking=True).float()

            out1, out2, out3, out4, out5, b1, b2, b3, b4, b5 = model(img, epoch, [0, 18])

            msk = np.asarray(msk.squeeze(), np.float32)
            res = out1
            res = F.interpolate(res, size=msk.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            # mIoU Calculation
            # binarized_res = (res > 0.5).astype(np.uint8)
            # binarized_msk = (msk > 0.5).astype(np.uint8)
            iou_sum += compute_iou(res, msk)

            # MAE Calculation
            mae_sum += np.sum(np.abs(res - msk)) * 1.0 / (msk.shape[0] * msk.shape[1])

        miou = iou_sum / len(test_loader)
        mae = mae_sum / len(test_loader)
        log_info = f'Epoch: {epoch} MAE: {mae:.4f} mIoU: {miou:.4f}'
        print(log_info)
        logger.info(log_info)

    return mae, miou

# def test_one_epoch(test_loader, model, criterion, logger, config, path, test_data_name=None):
#     model.eval()
#     gt_list = []
#     pred_list = []
#     total_miou = 0.0
#     total = 0
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(test_loader)):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
#
#             gt_pre, key_points, out = model(img)
#
#             msk = msk.squeeze().cpu().detach().numpy()
#             out = out.squeeze().cpu().detach().numpy()
#
#             gt_list.append(msk)
#             pred_list.append(out)
#
#             y_pre = np.where(out>=config.threshold, 1, 0)
#             y_true = np.where(msk>=0.5, 1, 0)
#
#             smooth = 1e-5
#             intersection = (y_pre & y_true).sum()
#             union = (y_pre | y_true).sum()
#             miou = (intersection + smooth) / (union + smooth)
#
#             total_miou += miou
#             total += 1
#
#             # if i % config.save_interval == 0:
#                 # kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12 = key_points
#                 # gt1, gt2, gt3, gt4, gt5 = gt_pre
#                 # save_imgs(img, msk, out, key_points, gt_pre, i, config.work_dir + 'outputs/' + 'ISIC2017' + '/', config.datasets, config.threshold, test_data_name=test_data_name)
#
#         total_miou = total_miou / total
#
#         pred_list = np.array(pred_list).reshape(-1)
#         gt_list = np.array(gt_list).reshape(-1)
#
#         y_pre = np.where(pred_list>=0.5, 1, 0)
#         y_true = np.where(gt_list>=0.5, 1, 0)
#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
#
#         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#
#         log_info = f'test of best model, miou: {total_miou}, f1_or_dsc: {f1_or_dsc}'
#         print(log_info)
#         logger.info(log_info)