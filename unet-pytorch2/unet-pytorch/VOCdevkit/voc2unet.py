import os
import random 
# F:\W_512\MyLIDC_5_数据处理\数据处理_test_2_2\dcm_1\unetpp32

def random_tra_val(dataset):

    segfilepath = os.path.join(dataset, "SegmentationClass")
    saveBasePath = os.path.join(dataset, "ImageSets\Segmentation")

    def del_file(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                del_file(c_path)
            else:
                os.remove(c_path)

    del_file(saveBasePath)

    trainval_percent = 1
    # 9:1
    train_percent = 0.8

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".txt"):
            total_seg.append(seg)

    num=len(total_seg)
    list=range(num)
    tv=int(num*trainval_percent)
    tr=int(tv*train_percent)
    trainval= random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4]+'\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest .close()


if __name__ == "__main__":
    dataset = r"D:\Code\Pycharm\unet-pytorch\unet-pytorch\dataset"
    random_tra_val(dataset)