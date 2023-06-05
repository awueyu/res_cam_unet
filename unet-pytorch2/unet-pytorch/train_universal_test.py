import os
import time

import ipdb
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from VOCdevkit.voc2unet import random_tra_val
from loss.focal_loss import FocalLoss
from nets.CAM_UNet import CAM_UNet

from nets.Dense_CAM_G_Ghsot_Half_UNet import Dense_CAM_G_Ghost_Half_UNet
from nets.ResNet18_CAM_UNet import Res18_CAM_UNet
from nets.Res_CAM_Unet import Res_CAM_Unet
from nets.unet_more import U_Net, CAM_U_Net
import opts
from nets.unet import Unet
import torch


from unet3plus.UNet_2Plus import UNet_2Plus
from unet3plus.UNet_3Plus import UNet_3Plus_Cam
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate, LossHistory
from utils.my_loss import BinaryDiceLoss, CE_Loss
from utils.my_utils import write_Args, model_info
from utils.utils import DiceLoss
torch.backends.cudnn.enabled = False
import warnings



warnings.filterwarnings("ignore")   # 忽略警告

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_loss_trval(masks_pred, pngs, true_masks):
    dice_loss_class = DiceLoss(NUM_CLASSES)
    binary_dice_loss = BinaryDiceLoss()
    focal_loss = FocalLoss()

    if args_opts.loss == "binary_dice_loss":
        loss = binary_dice_loss(masks_pred, true_masks)

    if args_opts.loss == "focal_loss":
        loss = focal_loss(masks_pred, pngs)

    if args_opts.loss == "ce_loss":
        loss = CE_Loss(masks_pred, pngs)

    if args_opts.loss == "ce_dice_loss":
        a = args_opts.loss_a
        loss = (1 - a) * CE_Loss(masks_pred, pngs) + a * binary_dice_loss(masks_pred, true_masks)

    return loss


def fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, genval, Epoch):
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0

    binary_dice_loss = BinaryDiceLoss()

    print("start training")
    start_time_train = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            # 2、backward
            start_time_step = time.time()
            imgs, pngs, true_masks = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                true_masks = torch.from_numpy(true_masks).type(torch.FloatTensor)

                imgs = imgs.cuda()
                pngs = pngs.cuda()
                true_masks = true_masks.cuda()

            # if args_opts.inputs_class == 3:
            #     if imgs.size()[1] == 1:
            #         imgs = imgs.repeat(1, 3, 1, 1)

            masks_pred = model(imgs).cuda()

            # 预测
            loss_o = get_loss_trval(masks_pred, pngs, true_masks)
            _f_score = 1 - binary_dice_loss(masks_pred, true_masks)

            if args_opts.use_add:
                # loss regularization  损失正规化
                loss = loss_o / cumulative_iters
                loss.backward()
                # 梯度累加策屡，判断是否达到执行梯度累积更新的条件
                if (iteration + 1) % cumulative_iters == 0:
                    # update parameters of net
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                loss = loss_o
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss_o.item()
            total_f_score += _f_score.item()

            step_waste_time = time.time() - start_time_step
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                's/step': step_waste_time,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

            # start_time = time.time()
    epoch_time_tain = time.time() - start_time_train
    global train_time_txts
    train_time_txts.append(epoch_time_tain)

    global train_lr
    for param_group in optimizer.param_groups:
        lr_epoch = param_group['lr']
        epoch_lr = str(epoch) + ": " + str(lr_epoch)
        train_lr.append(epoch_lr)

    print("train_time_opoch:", epoch_time_tain)
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            # imgs, true_masks = batch

            imgs, pngs, true_masks = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                true_masks = torch.from_numpy(true_masks).type(torch.FloatTensor)

                imgs = imgs.cuda()
                pngs = pngs.cuda()
                true_masks = true_masks.cuda()

                if args_opts.inputs_class == 3:
                    if imgs.size()[1] == 1:
                        imgs = imgs.repeat(1, 3, 1, 1)

                masks_pred = model(imgs).cuda()

                val_loss = get_loss_trval(masks_pred, pngs, true_masks)
                _f_score = 1 - binary_dice_loss(masks_pred, true_masks)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()
                val_f_score = val_total_f_score / (iteration + 1)
                epoch_val_loss = val_toal_loss / (iteration + 1)

            pbar.set_postfix(**{'total_loss': epoch_val_loss,
                                'f_score': val_f_score,
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    # 添加 相关的loss、valloss到 logs中，并绘图
    loss_history.append_loss(total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))

    if args_opts.lr_step is not True:
        if epoch == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        if epoch == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2

    global less_val_loss

    if epoch_val_loss < less_val_loss:
        less_val_loss = epoch_val_loss
        print('Saving state, iter:', str(epoch + 1))
        save_epoch = "Epoch%.3d-Total_Loss%.4f-Val_Loss%.4f.pth" % (
        (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1))
        save_path = os.path.join(pths_path, save_epoch)
        torch.save(model.state_dict(), save_path)
    # 保存全部  best_epoch_dice
    # if args_opts.loss is not "binary_dice_loss":
    #     global best_epoch_dice
    #     if val_f_score > best_epoch_dice:
    #         best_epoch_dice = val_f_score
    #         print('Saving state, iter:', str(epoch + 1))
    #         save_epoch = "Epoch%.3d-Total_Loss%.4f-Val_Loss%.4f.pth" % (
    #             (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1))
    #         save_path = os.path.join(all_pths_path, save_epoch)
    #         torch.save(model.state_dict(), save_path)




def return_model(my_model, inputs_class, NUM_CLASSES):


    if my_model == "Unet":
        model = Unet(in_ch=inputs_class, out_ch=NUM_CLASSES).cuda()

    if my_model == "U_Net":
        model = U_Net().cuda()

    if my_model == "UNet_2Plus":
        model = UNet_2Plus().cuda()

    if my_model == "CAM_UNet":
        model = CAM_UNet().cuda()

    if my_model == "UNet_3Plus_Cam":
        model = UNet_3Plus_Cam().cuda()

    if my_model == "Dense_CAM_G_Ghost_Half_UNet":
        model = Dense_CAM_G_Ghost_Half_UNet().cuda()


    if my_model == "Res18_CAM_UNet":
        model = Res18_CAM_UNet().cuda()

    if my_model == "Res_CAM_Unet":
        model = Res_CAM_Unet().cuda()
    return model


if __name__ == "__main__":

    args_opts = opts.parse_arguments()
    # 随机选取数据
    random_tra_val(args_opts.datadir)

    image_size_64_512 = args_opts.image_size
    NUM_CLASSES = args_opts.num_class
    inputs_class = args_opts.inputs_class

    cumulative_iters = args_opts.accu_steps
    torch.cuda.set_device(0)
    lr = args_opts.learning_rate
    Init_Epoch = 0
    Interval_Epoch = args_opts.epochs
    Batch_size = args_opts.batch_size
    out_path = args_opts.outdir
    dataset_path = args_opts.datadir

    pths_path = os.path.join(out_path, "1_val_loss")
    if not os.path.exists(pths_path):
        os.makedirs(pths_path)

    # if args_opts.loss is not "binary_dice_loss":
    #     all_pths_path = os.path.join(out_path, "2_val_loss")
    #     if not os.path.exists(all_pths_path):
    #         os.makedirs(all_pths_path)

    log_dir = os.path.join(out_path, "1_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    inputs_size = [image_size_64_512, image_size_64_512, inputs_class]
    # 配置文件
    write_Args(log_dir, args_opts)

    with open(os.path.join(dataset_path, "ImageSets\\Segmentation\\train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "ImageSets\\Segmentation\\val.txt"), "r") as f:
        val_lines = f.readlines()

    # 历史记录
    loss_history = LossHistory(log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    my_model = args_opts.model
    #  第 2 步构建模型
    model = return_model(my_model, inputs_class, NUM_CLASSES)

    # 相关参数量
    model_info(model, log_dir)
    # if args_opts.model is not "trans_unet":
    #     print("summary:")

    # 加载预训练权重
    if args_opts.load_weights:
        print("load_weights")
        model_path = args_opts.load_weights
        model_path_pth = os.path.join(out_path, model_path)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path_pth, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.train()
    optimizer = optim.Adam(model.parameters(), lr)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args_opts.lr_gamma)
    # 加载数据
    isjpg = args_opts.isjpg
    # #######################
    # 这里可以将 数据全部加载近来，然后再划分为 训练集，和 验证集
    train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True, dataset_path, isjpg=isjpg)
    val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False, dataset_path, isjpg=isjpg)

    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=args_opts.num_workers, pin_memory=True, shuffle=True,
                     drop_last=True, collate_fn=deeplab_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=args_opts.num_workers, pin_memory=True,
                         shuffle=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)

    epoch_size = len(train_lines) // Batch_size
    epoch_size_val = len(val_lines) // Batch_size

    if epoch_size == 0 or epoch_size_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    best_epoch_dice = 0
    less_val_loss = 1
    train_time_txts = []
    train_lr = []

    for epoch in range(Init_Epoch, Interval_Epoch):
        start_time_epoch = time.time()

        fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Interval_Epoch)
        if args_opts.lr_step:
            lr_scheduler.step()

        epoch_time = time.time() - start_time_epoch
        print("每轮所需时间：", epoch_time)
        # print("", lr_scheduler.get_lr())

    train_txts_path = os.path.join(log_dir, "tarin_time.txt")
    with open(train_txts_path, "a") as time_t:
        for i in range(len(train_time_txts)):
            time_t.write(str(train_lr[i]))
            time_t.write('\n')
    with open(train_txts_path, "a") as time_t:
        for i in range(len(train_time_txts)):
            time_t.write(str(train_time_txts[i]))
            time_t.write('\n')