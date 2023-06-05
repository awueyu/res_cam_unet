# 可以得到中间的 特征图
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2


# model = torch.load('./save/model.pkl').to(torch.device('cpu'))
import opts
from train_universal_test import return_model

# 加载 训练好的 模型
args_opts = opts.parse_arguments()
my_model = args_opts.model
model = return_model(my_model, 3, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

model_path = r"F:\0_LIDC\new_512\voc_512_lung_result\U_Net32\1_val_loss\Epoch051-Total_Loss0.1905-Val_Loss0.2071.pth"
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model = model.to(device)


# model = models.resnet50(pretrained=True)
print(model)

# 从测试集中读取一张图片，并显示出来
img_path = r"F:\0_LIDC\new_512\voc_512_lung_result\U_Net32\P243-0000.jpg"
# img = Image.open(img_path)
# imgarray = np.array(img) / 255.0

# plt.figure(figsize=(8, 8))
# plt.imshow(imgarray)
# plt.axis('off')
# plt.show()

# 将图片处理成模型可以预测的形式
transform = transforms.Compose([
    # transforms.Resize([224, 224]),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# input_img = transform(img).unsqueeze(0)

image = Image.open(img_path)
image = np.array(image, dtype='float32')
images = image.reshape([512, 512, -1])
images = np.transpose(np.array(images), [2, 0, 1])
images = torch.from_numpy(images).type(torch.FloatTensor)
images = images.cuda()
input_img = images[None, :, :, :]

print(input_img.shape)


# 定义钩子函数，获取指定层名称的特征
activation = {} # 保存获取的输出

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.eval()
# 获取layer1里面的bn3层的结果，浅层特征
model.Up_conv2.conv.register_forward_hook(get_activation('4'))
# model.layer1[0].register_forward_hook(get_activation('bn3'))

# model_layer= list(model.children())
# print(model_layer)
# model_layer=model_layer[0]

# 为layer1中第2个模块的bn3注册钩子
_ = model(input_img)

bn3 = activation['4'].cpu() # 结果将保存在activation字典中
print(bn3.shape)


savedir = r'F:\0_LIDC\new_512\voc_512_lung_result\U_Net32'
if not os.path.exists(savedir + '\\val_pred_temp'):
    os.mkdir(savedir + '\\val_pred_temp')
for i in range(32):
    outlay_path = savedir + '\\val_pred_temp\\' + str(i) + '.jpg'
    plt.imsave(outlay_path, bn3[0, i, :, :], cmap='gray')


# 可视化结果，显示前64张
# plt.figure(figsize=(12, 12))
# for i in range(64):
#     plt.subplot(8, 8, i+1)
#     plt.imshow(bn3[0, i, :, :], cmap='gray')
#     plt.axis('off')
# plt.show()


class GradCAM(nn.Module):
    def __init__(self):
        super(GradCAM, self).__init__()
        # 获取模型的特征提取层
        self.feature = nn.Sequential(OrderedDict({
            name: layer for name, layer in model.named_children()
            if name not in ['1conv10', 'fc', 'Sigmoid']
        }))
        # 获取模型最后的平均池化层
        # self.conv10 = model.conv10
        # 获取模型的输出层
        # self.classifier = nn.Sequential(OrderedDict([
        #     # ('fc', model.fc),
        #     ('Sigmoid', model.Sigmoid)
        # ]))
        # 生成梯度占位符
        self.gradients = None

    # 获取梯度的钩子函数
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.feature(x)
        # 注册钩子
        h = x.register_hook(self.activations_hook)
        # 对卷积后的输出使用平均池化
        # x = self.conv10(x)
        x = nn.Sigmoid()(x)
        # x = x.view((1, -1))
        # x = self.classifier(x)
        return x

    # 获取梯度的方法
    def get_activations_gradient(self):
        return self.gradients

    # 获取卷积层输出的方法
    def get_activations(self, x):
        return self.feature(x)

# 获取热力图
def get_heatmap(model, img):
    model.eval()
    img_pre = model(img)
    # 获取预测最高的类别
    pre_class = torch.argmax(img_pre, dim=-1).item()
    # 获取相对于模型参数的输出梯度
    img_pre[:, pre_class].backward()
    # 获取模型的梯度
    gradients = model.get_activations_gradient()
    # 计算梯度相应通道的均值
    mean_gradients = torch.mean(gradients, dim=[0,2,3])
    # 获取图像在相应卷积层输出的卷积特征
    activations = model.get_activations(input_img).detach()
    # 每个通道乘以相应的梯度均值
    for i in range(len(mean_gradients)):
        activations[:,i,:,:] *= mean_gradients[i]
    # 计算所有通道的均值输出得到热力图
    heatmap = torch.mean(activations, dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

# cam = GradCAM()
# # 获取热力图
# heatmap = get_heatmap(cam, input_img)
#
#
# # 可视化热力图
# plt.matshow(heatmap)
# plt.show()