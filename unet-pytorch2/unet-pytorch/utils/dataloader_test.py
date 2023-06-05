import os
import random
from random import shuffle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.signal
import torch
import torchvision
from torch.nn import functional
from PIL import Image
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class DeeplabDataset(Dataset):
    # inputs_size = [64, 64, 1]
    # DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, False, dataset_path)
    def __init__(self, train_lines, image_size, num_classes, random_data, dataset_path, isjpg=True):
        super(DeeplabDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        self.random_data = random_data
        self.dataset_path = dataset_path
        self.isjpg = isjpg
        # self.totensor = torchvision.transforms.ToTensor()

        # 将其全部放在内存中
        self.jpg_all = []
        self.mask_all = []
        for annotation_line in train_lines:
            name = annotation_line[:9]

            if self.isjpg == True:
                image = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
                image = np.array(image, dtype="float32")
                # image = cv2.resize(image, (256, 256), interpolation=cv2.CV_INTER_AREA)
                self.image_height, self.image_width, c = image.shape
            else:
                # 从文件中读取图像
                plan = pydicom.read_file(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".dcm"))
                image = plan.pixel_array
                image = np.array(image, dtype="float32")
                self.image_height, self.image_width = image.shape

            # 加载肺结节轮廓坐标
            txt_file = os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".txt")
            x, y = np.loadtxt(txt_file).T
            mask = self.contour_to_mask(x, y, norm=1)
            self.jpg_all.append(image)
            self.mask_all.append(mask)

    # len(dataset)返回整个数据集的大小
    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def contour_to_mask(self, x, y, norm=1):
        BW_8BIT = 'L'
        # zip :打包为元组的列表,而且元素个数与最短的列表一致 [(x1, y1), (x2, y2)...]
        polygon = list(zip(x, y))
        image_dims = (self.image_width, self.image_height)
        # Image.new(mode, size, color=0) :是用来new一个新的图像
        # mode:模式, color：生成图像的颜色，默认为0，即黑色
        # mode: "1-二值图"，“L-灰度图”，“RGB-三原色”"RGBA-三原色+透明度alpha"
        img = Image.new(BW_8BIT, image_dims, color=0)

        # 定义：Draw(image) ⇒ Draw instance
        # 含义：创建一个可以在给定图像上绘图的对象。  ImageDraw.Draw(img) 就是创建一个对象

        # polygon 英 [ˈpɒlɪɡən] n. 多边形；多角形物体
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        # uint8是专门用于存储各种图像的(包括RGB，灰度图像等)，范围是从0–255
        return norm * np.array(img, dtype='uint8')

    # __getitem__用来获取一些索引的数据，使dataset[i]返回数据集中第i个样本

    def __getitem__(self, index):
        if index == 0:
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(self.jpg_all)
            random.seed(randnum)
            random.shuffle(self.mask_all)

        image = self.jpg_all[index]
        mask = self.mask_all[index]

        png = mask

        image = image.reshape([self.image_height, self.image_height, -1])
        image = np.transpose(np.array(image), [2, 0, 1])

        mask = mask.reshape([self.image_height, self.image_height, -1])
        mask = np.transpose(np.array(mask), [2, 0, 1])

        return image, png, mask


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels


class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth= 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))
