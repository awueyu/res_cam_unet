from __future__ import division, print_function

import argparse
import configparser
import logging

definitions = [

    ('model',            (str,   'Res_CAM_Unet',     "Model: unet  cam_unet  ghost_unet  deform_unet  trans_unet       ")),
    ('epochs',           (int,   60,        "Number of epochs to train.")),
    ('batch-size',       (int,   8,         "Mini-batch size for training.")),
    ('loss',             (str,   'binary_dice_loss',    "Loss function:  focal_binary    binary_dice_loss   ce_dice  BinaryDiceLoss   ce_dice_loss ")),
    ('loss_a',           (float, 0.5,    " (1 - a) * focal_loss(masks_pred, true_masks)")),
    ('outdir',           (str,   r'D:\Code\Pycharm\unet-pytorch\Res_CAM_Unet',   "Directory to write output data.")),
    ('load-weights',     (str,   r'',          "Load model weights")),
    ('datadir',          (str,   r'D:\Code\Pycharm\unet-pytorch\unet-pytorch\dataset',    "Directory containing patientXX/ directories.")),
    ('isjpg',            {'default': True,  'help': "if true , step lr"}),
    ('num_workers',      (int,   1,           "num_workers.")),
    ('lr_step',          {'default': True,  'help': "if true , step lr"}),
    ('lr_gamma',         (float,   0.99,           "lr_gam"
                                                   "ma")),
    ('num_class',        (int,   1,           "lr_gamma")),
    ('inputs_class',     (int,   1,           "lr_gamma")),
    ('image_size',       (int,   128,           "lr_gamma")),
    ('accu_steps',       (int,   8,           "梯度累加迭代的次数")),
    ('use_add',          {'default': False,  'help': "if true 使用梯度累加策略"}),
    ('learning-rate',    (float, 0.001,   "Optimizer learning rate.")),


    # #####################################

    ('features',         (int,   512,     "Number of features maps after first convolutional layer.")),
    ('depth',            (int,   4,      "Number of downsampled convolutional blocks.")),
    ('temperature',      (float, 1.0,    "Temperature of final softmax layer in model.")),
    ('padding',          (str,   'same', "Padding in convolutional layers. Either `same' or `valid'.")),
    ('dropout',          (float, 0,    "Rate for dropout of activation units.")),
    ('batchnorm',        {'default': True, 'action': 'store_true',
                          'help': "Apply batch normalization before nonlinearities."}),
    # loss
    ('loss-weights',     {'type': float, 'nargs': '+', 'default': [0.1, 0.9],
                          'help': "When using dice or jaccard loss, how much to weight each output class."}),
    # training
    ('validation-split', (float, 0.3,    "Percentage of training data to hold out for validation.")),
    ('optimizer',        (str,   'adam', "Optimizer: sgd, rmsprop, adagrad, adadelta, adam, adamax, or nadam.")),

    ('momentum',         (float, None,   "Momentum for SGD optimizer.")),
    ('decay',            (float, None,   "Learning rate decay (not applicable for nadam).")),
    ('shuffle_train_val', {'default': True, 'action': 'store_true',
                           'help': "Shuffle images before splitting into train vs. val."}),
    ('shuffle',          {'default': False, 'action': 'store_true',
                          'help': "Shuffle images before each training epoch. 在每个训练epoch 前 洗牌图像"}),
    ('seed',             (int,   None,   "Seed for numpy RandomState")),
    ('outfile',          (str,   'weights-final.hdf5', "File to write final momonitordel weights.")),
    ('checkpoint',       {'default': True, 'action': 'store_true',
                          'help': "Write model weights after each epoch if validation accuracy improves."}),
    ('augment-training', {'default': False, 'action': 'store_true',
                          'help': "Whether to apply image augmentation to training set."}),
    ('augment-validation', {'default': False, 'action': 'store_true',
                            'help': "Whether to apply image augmentation to validation set."}),
    ('rotation-range',     (float, 180,    "Rotation range (0-180 degrees)")),
    ('width-shift-range',  (float, 0.1,    "Width shift range, as a float fraction of the width, 宽度移位范围，作为宽度的浮点分数")),
    ('height-shift-range', (float, 0.1,    "Height shift range, as a float fraction of the height")),
    ('shear-range',        (float, 0.1,    "Shear intensity (in radians), 剪切强度(以弧度为单位)")),
    # 变焦。如果是标量z，放大[1-z, 1+z]。也可以通过一对浮动作为变焦范围。
    ('zoom-range',         (float, 0.05,   "Amount of zoom. If a scalar z, zoom in [1-z, 1+z]. Can also pass a pair of floats as the zoom range.")),
    # 边界外的点根据模式填充:常量、最近点、反射点或包裹点
    ('fill-mode',          (str,   'nearest', "Points outside boundaries are filled according to mode: constant, nearest, reflect, or wrap")),
    ('alpha',              (float, 500,    "Random elastic distortion: magnitude of distortion")),
    ('sigma',              (float, 20,     "Random elastic distortion: length scale")),
    ('normalize', {'default': False, 'action': 'store_true',
                   'help': "Subtract mean and divide by std dev from each image. 从每个图像中减去mean并除以 std dev"}),
]


noninitialized = {
    'learning_rate': 'getfloat',
    'momentum': 'getfloat',
    'decay': 'getfloat',
    'seed': 'getint',
}


def update_from_configfile(args, default, config, section, key):
    # Point of this function is to update the args Namespace.
    value = config.get(section, key)
    if value == '' or value is None:
        return

    # Command-line arguments override config file values
    if getattr(args, key) != default:
        return

    # Config files always store values as strings -- get correct type
    if isinstance(default, bool):
        value = config.getboolean(section, key)
    elif isinstance(default, int):
        value = config.getint(section, key)
    elif isinstance(default, float):
        value = config.getfloat(section, key)
    elif isinstance(default, str):
        value = config.get(section, key)
    elif isinstance(default, list):
        # special case (HACK): loss-weights is list of floats
        string = config.get(section, key)
        value = [float(x) for x in string.split()]
    elif default is None:
        # values which aren't initialized
        getter = getattr(config, noninitialized[key])
        value = getter(section, key)
    setattr(args, key, value)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train U-Net to segment right ventricles from cardiac "
        "MRI images.")

    for argname, kwargs in definitions:
        d = kwargs
        if isinstance(kwargs, tuple):
            d = dict(zip(['type', 'default', 'help'], kwargs))
        parser.add_argument('--' + argname, **d)

    # allow user to input configuration file
    parser.add_argument(
        'configfile', nargs='?', type=str, help="Load options from config "
        "file (command line arguments take precedence).")

    args = parser.parse_args()

    if args.configfile:
        logging.info("Loading options from config file: {}".format(args.configfile))
        config = configparser.ConfigParser(
            inline_comment_prefixes=['#', ';'], allow_no_value=True)
        config.read(args.configfile)
        for section in config:
            for key in config[section]:
                if key not in args:
                    raise Exception("Unknown option {} in config file.".format(key))
                update_from_configfile(args, parser.get_default(key),
                                       config, section, key)

    for k, v in vars(args).items():
        logging.info("{:20s} = {}".format(k, v))

    return args
