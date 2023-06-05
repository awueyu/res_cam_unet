import PIL.Image
import pydicom
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import opts
import os
import numpy as np
from skimage import morphology, measure
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from train_universal_test import return_model


def get_contours_best(mask):
    # np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
    mask_image = np.where(mask > 0.5, 255, 0).astype('uint8')

    labeled_img, num = measure.label(mask_image, neighbors=4, background=0, return_num=True)
    props = measure.regionprops(labeled_img)
    # 在连通区域中找 maxarea 的
    maxarea = 0
    num = 0
    # array([[0, 0],[0, 0]])  shape:{tuple 2} (2, 2)
    contour = np.zeros((2, 2), dtype=np.int8)
    for region in props:
        # area -- 区域内像素点总数
        if region.area > maxarea:
            maxarea = region.area
            num = region.label
            # coords -- ndarray	-- 区域内像素点坐标
            contour = region.coords

    img = np.zeros((image_size_64_512, image_size_64_512))
    img[contour[:, 0], contour[:, 1]] = 1

    # numpy.clip(m, min, max)
    # 功能：把数组 m 中的值缩放到 [min, max] 之间
    img = np.clip(img, 0, image_size_64_512)
    img = np.array(img, np.uint8)

    chull = morphology.convex_hull_object(img)
    hullimage = np.array(chull, np.uint8)
    ppp = np.sum(hullimage)

    contours, hierarchy = cv2.findContours(hullimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    coords_prival = np.squeeze(contours, axis=2)
    coords = coords_prival[0, :, :]
    return coords


def get_contours(mask):
    mask_image = np.where(mask > 0.5, 255, 0).astype('uint8')

    chull = morphology.convex_hull_object(mask_image)
    hullimage = np.array(chull, np.uint8)

    contours, hierarchy = cv2.findContours(hullimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if hasattr(hierarchy, 'shape'):
        h_shape = hierarchy.shape[1]
        coords = []
        for i in range(0, h_shape):
            coords_prival = np.squeeze(contours[i], axis=1)
            coords.extend(coords_prival[:, :])
        coords = np.array(coords)
    else:
        coords = [[0, 0]]
        coords = np.array(coords)
    return coords


def detect_image(images,name,image_path):
    im = images
    with torch.no_grad():
        images = images.reshape([image_size_64_512, image_size_64_512, -1])
        images = np.transpose(np.array(images), [2, 0, 1])

        images = torch.from_numpy(images).type(torch.FloatTensor)
        images = images.cuda()

        imgs = images[None, :, :, :]
        if args_opts.inputs_class == 3:
            if imgs.size()[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

        mask_pred_s = model(imgs)
        # mask_pred_s = F.softmax(mask_pred, dim=1).float()
        mask_pred_s = mask_pred_s.cpu()
        # mask_pred = np.transpose(np.array(mask_pred), [0, 2, 3, 1])
        mask = mask_pred_s[0, 0, :, :]
        # mask_image = np.where(mask_pred_s[0, 0, :, :] > 0.5, 255, 0).astype('uint8')
        contour = get_contours(mask)

        im = cv2.imread(image_path)
        cv2.polylines(im, np.array([contour]), True, [0, 0, 255], 1)
        PIL.Image.fromarray(im).save(r"D:\Code\Pycharm\unet-pytorch\test\res_{}.jpg".format(name))

        # contour = get_contours_best(mask)
    return contour


if __name__ == "__main__":
    torch.cuda.set_device(0)
    args_opts = opts.parse_arguments()
    image_size_64_512 = args_opts.image_size
    NUM_CLASSES = args_opts.num_class
    inputs_class = args_opts.inputs_class
    out_path = args_opts.outdir
    # pths  best_valdice  1_fianal  1_val_loss
    pths_path = os.path.join(out_path, "1_val_loss")
    # for root, dirs, files in os.walk(pths_path):
    # print(root)
    if args_opts.load_weights:
        filename_pth = args_opts.load_weights
    else:
        list_pths = os.listdir(pths_path)
        # 如果不存在，选取最后一个
        filename_pth = list_pths[-1]
    print(filename_pth)
    model_path = os.path.join(pths_path, filename_pth)
    new_file_path = os.path.join(out_path, filename_pth[0:9])
    # hello
    outdir_txts = os.path.join(out_path, "test_txt")
    if not os.path.exists(outdir_txts):
        os.makedirs(outdir_txts)
    # if not os.path.exists(new_file_path):
    #     os.makedirs(new_file_path)
    inputs_size = [image_size_64_512, image_size_64_512, inputs_class]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #0.893854
    my_model = args_opts.model
    #  第 2 步构建模型
    model = return_model(my_model, inputs_class, NUM_CLASSES)

    model = model.eval().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    dataset_path = args_opts.datadir
    test_txt = os.path.join(dataset_path, "ImageSets\\test.txt")
    image_ids = open(test_txt, 'r').read().splitlines()
    for image_id in tqdm(image_ids):
        name = image_id[:9]

        if args_opts.isjpg:
            dcm_path = r"test_dcm_center\\" + name + ".jpg"
            image_path = os.path.join(dataset_path, dcm_path)
            image = Image.open(image_path)
        else:
            # dcm
            dcm_path = r"test_dcm_center\\" + name + ".dcm"
            image_path = os.path.join(dataset_path, dcm_path)
            plan = pydicom.read_file(image_path)
            image = plan.pixel_array

        image = np.array(image, dtype='float32')

        # image = Image.open(image_path)
        contour = detect_image(image,name,image_path)

        filename = "{}.txt".format(name)
        outpath = os.path.join(outdir_txts, filename)
        np.savetxt(outpath, contour, fmt='%i', delimiter=' ')
