import os
from pytorch_grad_cam import *

ROOT_DIR = "/home/r6639/Projects/xaiseg/data/pascal_voc_2012"
DATA_DIR = os.path.join(ROOT_DIR, "ds")
ANN_DIR = os.path.join(DATA_DIR, "ann")
IMG_DIR = os.path.join(DATA_DIR, "img")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
VAL_DIR = os.path.join(ROOT_DIR, "val")
INFERENCE_DIR = os.path.join(ROOT_DIR, "inference")
x_train_dir = os.path.join(TRAIN_DIR, 'img')
y_train_dir = os.path.join(TRAIN_DIR, 'ann')
x_test_dir = os.path.join(TEST_DIR, 'img')
y_test_dir = os.path.join(TEST_DIR, 'ann')
x_val_dir = os.path.join(VAL_DIR, 'img')
y_val_dir = os.path.join(VAL_DIR, 'ann')

EPOCHS = 1000

# Define your classes here, for example:
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'neutral', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

ENCODER = 'resnet101'
ENCODER_STUDENT = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATIONS = 'softmax2d'

XAI_METHODS = [GradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM, ScoreCAM, HiResCAM, AblationCAM, XGradCAM, LayerCAM,
               FullGrad]
