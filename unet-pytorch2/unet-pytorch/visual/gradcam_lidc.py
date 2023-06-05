import os
import warnings

import cv2
from matplotlib import pyplot as plt

import opts
from train_universal_test import return_model
import torch.nn as nn
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
# from utils_gradcam import GradCAM, show_cam_on_image
from pytorch_grad_cam.utils.image import show_cam_on_image


def show_cam_on_image_me(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap * 0.5 + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_grad_cam(img_path, model, target_layers, out_path):
    # img_path = "P478-0000.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    image1 = np.array(Image.open(img_path))

    rgb_img = np.float32(image1) / 255

    image = Image.open(img_path)
    image = np.array(image, dtype='float32')
    images = image.reshape([512, 512, -1])
    images = np.transpose(np.array(images), [2, 0, 1])
    images = torch.from_numpy(images).type(torch.FloatTensor)
    images = images.cuda()
    input_tensor = images[None, :, :, :]


    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    output = model(input_tensor)

    # 第三部分
    normalized_masks = output.cpu()

    car_category = 1
    # car_mask 这个是轿车 这一类的 mask， 里边的值为 7
    car_mask = np.where(normalized_masks[0, 0, :, :] > 0.5, 1, 0)
    # 这个将 7 变为 255
    car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
    # float 中的 7 的位置 变为1.000
    car_mask_float = np.float32(car_mask == car_category)

    # (h, w, 3)
    both_images = np.hstack((image1, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
    Image.fromarray(both_images)
    # plt.imshow(both_images)
    # plt.show()

    # 最后一部分
    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()

        def __call__(self, model_output):
            return (model_output[self.category, :, :] * self.mask).sum()


    # target_layers = [model.conv9]
    # 这个是某个类别的？
    targets = [SemanticSegmentationTarget(0, car_mask_float)]

    # 1.3.9
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    # 这个自己调用的有
    cam_image = show_cam_on_image_me(rgb_img, grayscale_cam, use_rgb=True)


    outImg = Image.fromarray(cam_image)
    outImg = outImg.resize((300, 300))
    outImg.save(out_path, quality=100, dpi=(100.0, 100.0))


if __name__ == "__main__":
    #

    base_path = r"F:\0_LIDC\new_512\VOC_512_result\MobileNetV3Unet"
    # 原图片的位置  D:\dataset_2022_64\VOC2007_64_567
    jpg_all_path = r"F:\0_LIDC\new_512\VOC_512\test_dcm_center"
    # 获取模型
    my_model = "MobileNetV3Unet"
    model = return_model(my_model, 3, 1)
    model_path_1_val_loss = os.path.join(base_path, "1_val_loss")
    list_pths = os.listdir(model_path_1_val_loss)
    filename_pth = list_pths[-1]
    model_path = os.path.join(model_path_1_val_loss, filename_pth)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.eval()
    print(model)

    # 各种参数
    # img_path = "P478-0000.jpg"
    # model.swin_unet.decoder.conv9
    # model.Up_conv2.conv
    target_layers = [model.conv9.conv]
    # target_layers = [model.conv9]
    # out_path = r"F:\dataset_2022_64_result\8910jpg311\unet\P478-00.jpg"

    #
    out_all_path = os.path.join(base_path, "out_grad_cam-5")
    if not os.path.exists(out_all_path):
        os.makedirs(out_all_path)

    jpg_lists = os.listdir(jpg_all_path)
    for j_path in jpg_lists:
        print(j_path)
        img_path = os.path.join(jpg_all_path, j_path)
        out_path = os.path.join(out_all_path, j_path)

        get_grad_cam(img_path, model, target_layers, out_path)