import sys
sys.path.insert(0, './')
import torch
from torch.utils.data import DataLoader
import timm
from dataset import NPY_datasets
from decoder import self_net
from engine import *
from tensorboardX import SummaryWriter
import os
from utils import *
from config_setting import setting_config
import pytorch_ssim
from Loss import CEL, IOU

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir)
    sys.path.append(config.work_dir)
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    # resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    # outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # if not os.path.exists(outputs):
    #     os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)



    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()



    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=8,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=4,
                                drop_last=True)
    print('train_loader:', len(train_loader))
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0,
                                drop_last=False)
    print('val_loader:', len(val_loader))



    print('#----------Preparing Model----------#')
    if config.network == 'RailNet':
        model = self_net()
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    print('#----------Preparing loss, opt, sch and amp----------#')
    bce_loss = nn.BCELoss(size_average=True)
    ssim_loss = pytorch_ssim.SSIM(window_size=7, size_average=True)
    iou_loss = IOU(size_average=True)


    def hybrid_loss(pred, target):
        bce_out = bce_loss(pred, target)
        # ssim_out = 1 - ssim_loss(pred, target)
        iou_out = iou_loss(pred, target)
        loss = bce_out + iou_out
        return loss

    def hybrid_loss2(pred, target):
        bce_out = bce_loss(pred, target)
        # ssim_out = 1 - ssim_loss(pred, target)
        ssim_out = 1 - ssim_loss(pred, target)
        loss = bce_out + ssim_out
        return loss

    def multi_loss_function(r1, r2, r3, r4, r5, b1, b2, b3, b4, b5, label_r, label_b):

        loss1 = hybrid_loss(r1, label_r) + 0.5 * hybrid_loss2(b1, label_b)
        loss2 = hybrid_loss(r2, label_r) + 0.5 * hybrid_loss2(b2, label_b)
        loss3 = hybrid_loss(r3, label_r) + 0.5 * hybrid_loss2(b3, label_b)
        loss4 = hybrid_loss(r4, label_r) + 0.5 * hybrid_loss2(b4, label_b)
        loss5 = hybrid_loss(r5, label_r) + 0.5 * hybrid_loss2(b5, label_b)

        loss = loss1 + 0.9 * loss2 + 0.8 * loss3 + 0.6 * loss4 + 0.5 * loss5

        return loss

    criterion = multi_loss_function
    optimizer = get_optimizer(config, model)
    print(optimizer)
    scheduler = get_scheduler(config, optimizer)
    print(scheduler)


    print('#----------Set other params----------#')
    min_value = 999
    max_iou = 0
    start_epoch = 1
    min_epoch = 1


    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        if epoch % 100 == 0:  # save model every 1000 iterations
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'train_epoch_{epoch}.pth'))

        value, miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
        )

        if value < min_value:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_mae.pth'))
            min_value = value
            min_epoch = epoch
        if miou > max_iou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_miou.pth'))
            max_iou = miou


if __name__ == '__main__':
    config = setting_config
    main(config)