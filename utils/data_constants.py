import json

IMAGE_TASKS = ['rgb', 'depth', 'ired', 'sired', 'ebands', 'semseg', 'semsegdw', 'semseg_coco']
file_path = 'data_1M_v001_64_band_stats.json' #Change me
norm_values = json.load(open(file_path))

DEPTH_MEAN = norm_values['aster']['mean']
DEPTH_STD = norm_values['aster']['std']

S2_MEAN = norm_values['sentinel2_l2a']['mean']
S2_STD = norm_values['sentinel2_l2a']['std']

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

CIFAR_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_DEFAULT_STD = (0.2023, 0.1994, 0.2010)

SEG_IGNORE_INDEX = 255
PAD_MASK_VALUE = 254
COCO_SEMSEG_NUM_CLASSES = 133
MMEARTH_SEMSEG_NUM_CLASSES = 12

NYU_MEAN = 2070.7764
NYU_STD = 777.5723