import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix

import numpy as np
import torch
import torch.distributed as dist
import time
from tqdm import tqdm
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    pass
import os

def train_model(
    trainLoader, model, epoch, epoch_number, optimizer, combiner, criterion, cfg, logger, rank=0, use_apex=True, **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    trainLoader.dataset.update(epoch)
    combiner.update(epoch)
    criterion.update(epoch)


    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        loss, now_acc = combiner.forward(model, criterion, image, label, meta)

        optimizer.zero_grad()

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, distributed, **kwargs
):
    model.eval()

    if cfg.LOSS.LOSS_TYPE=="DiVEKLD":
        criterion = criterion.base_loss
    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()

        func = torch.nn.Sigmoid() \
            if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
            torch.nn.Softmax(dim=1)

        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True, label=label)
            loss = criterion(output, label, feature=feature)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())

            if distributed:
                world_size = float(os.environ.get("WORLD_SIZE", 1))
                reduced_loss = reduce_tensor(loss.data, world_size)
                reduced_acc = reduce_tensor(torch.from_numpy(np.array([acc])).cuda(), world_size)
                loss = reduced_loss.cpu().data
                acc = reduced_acc.cpu().data

            all_loss.update(loss.data.item(), label.shape[0])
            if distributed:
                acc_avg.update(acc.data.item(), cnt*world_size)
            else:
                acc_avg.update(acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc_avg.avg * 100
        )
        if rank == 0:
            logger.info(pbar_str)
    return acc_avg.avg, all_loss.avg
