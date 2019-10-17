import os
import sys
import argparse

from lane_change_risk_detection.dataset import *
from lane_change_risk_detection.models import Models
from keras.applications import ResNet50
from keras.models import Model
import pandas as pd


def set_dataset(image_path, label_path):

    df = pd.read_csv(label_path, header=0, usecols=[3, 4])

    target_data = np.zeros([len(df['no_event'].tolist()), 2])
    target_data[:, 0] = df['no_event'].tolist()
    target_data[:, 1] = df['critical'].tolist()

    backbone_model = ResNet50(weights='imagenet')
    backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.get_layer(index=-2).output)

    data = DataSet()
    data.model = backbone_model
    data.extract_features(image_path, option='fixed frame amount', number_of_frames=50)

    data.risk_one_hot = target_data

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--training_data_dir", dest='train_image_path', help="training data directory", type=str,
                        default="/media/ekim-hpc/hdd1/data/holistic_driving_risk/train")
    parser.add_argument("-v", "--validation_data_dir", dest='val_image_path', help="validation data directory", type=str,
                        default="/media/ekim-hpc/hdd1/data/holistic_driving_risk/validation")
    parser.add_argument("-e", "--nb_epoch", dest='nb_epoch', help="number of epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", dest='batch_size', help="batch size", type=int, default=32)
    parser.add_argument("-m", "--model_name", dest='model_name', help="name of the model to be trained", type=str, default='new_model.h5')

    args = parser.parse_args()

    train_data = set_dataset(args.train_image_path, os.path.join(args.train_image_path, 'train.csv'))
    val_data = set_dataset(args.val_image_path, os.path.join(args.val_image_path, 'validation.csv'))

    train_data.save(filename='mit_resnet_train.pickle', save_dir=args.train_image_path)
    val_data.save(filename='mit_resnet_val.pickle', save_dir=args.val_image_path)

    train_x = train_data.video_features
    train_y = train_data.risk_one_hot
    val_x = val_data.video_features
    val_y = val_data.risk_one_hot

    model = Models(nb_epoch=args.nb_epoch, batch_size=args.batch_size)
    model.build_transfer_LSTM_model(input_shape=train_x.shape[1:])
    model.model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=args.batch_size, nb_epoch=args.nb_epoch, verbose=2)
    model.model.save(args.model_name)

