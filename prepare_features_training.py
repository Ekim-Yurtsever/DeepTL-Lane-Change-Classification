from __future__ import print_function
from dataset import *
import cv2
import numpy as np

import keras
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, CuDNNGRU
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import roc_auc_score

from tensorflow.python.client import device_lib

from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import Model
import matplotlib.pyplot as plt
import random


#print(device_lib.list_local_devices())

data = DataSet()
data.read_features(feature_size=2048, feature_path='/media/ekim-hpc/hdd1/lane_change_risk_detection/extracted_features/res_net_50_imagenet', number_of_frames=50)
data.read_risk_data("LCTable.csv")
data.convert_risk_to_one_hot(risk_threshold=0.05)
#data.decode_one_hot()
#data.read_video("/media/ekim-hpc/hdd1/lane_change_risk_detection/rescaled_images_3e-1", scaling='no scaling')
data.save(save_dir='/media/ekim-hpc/hdd1/lane_change_risk_detection/saved data/', filename='dataset_50frames_resnet_features_5percent.pickle')