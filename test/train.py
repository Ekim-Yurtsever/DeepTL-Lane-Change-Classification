import os
import sys
import argparse

from lane_change_risk_detection.dataset import *
from lane_change_risk_detection.models import Models
from keras.applications import ResNet50
from keras.models import Model
import pandas as pd


def set_dataset(image_path, label_path, feature_extract_option=0, feature_path='/mit_resnet_train.pickle'):

    df = pd.read_csv(label_path, header=0, usecols=[3, 4])

    target_data = np.zeros([len(df['no_event'].tolist()), 2])
    target_data[:, 0] = df['no_event'].tolist()
    target_data[:, 1] = df['critical'].tolist()

    data = DataSet()
    data.risk_one_hot = target_data

    if feature_extract_option == 0:
        backbone_model = ResNet50(weights='imagenet')
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.get_layer(index=-2).output)
        data.model = backbone_model
        data.extract_features(image_path, option='fixed frame amount', number_of_frames=190)
    elif feature_extract_option == 1:
        data.video_features = DataSet.loader(image_path + feature_path)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--training_data_dir", dest='train_image_path', help="training data directory", type=str,
                        default="/media/ekim-hpc/hdd1/data/holistic_driving_risk/train")
    parser.add_argument("-v", "--validation_data_dir", dest='val_image_path', help="validation data directory", type=str,
                        default="/media/ekim-hpc/hdd1/data/holistic_driving_risk/validation")
    parser.add_argument("-e", "--nb_epoch", dest='nb_epoch', help="number of epochs", type=int, default=500)
    parser.add_argument("-b", "--batch_size", dest='batch_size', help="batch size", type=int, default=32)
    parser.add_argument("-m", "--model_name", dest='model_name', help="name of the model to be trained", type=str, default='new_model.h5')
    parser.add_argument("-d", "--dataset_option", dest='data_option', help="read and extract =0 load=1 ", type=int,
                        default=1)
    parser.add_argument("-s", "--dataset_save_option", dest='data_save_option', help="save the dataset as pickle=0 dont save=1 ", type=int,
                        default=1)

    args = parser.parse_args()

    if args.data_option == 0:
        train_data = set_dataset(args.train_image_path, os.path.join(args.train_image_path, 'train.csv'), feature_extract_option=0, feature_path='/mit_resnet_train.pickle')
        val_data = set_dataset(args.val_image_path, os.path.join(args.val_image_path, 'validation.csv'), feature_extract_option=0, feature_path='/mit_resnet_validation.pickle')
    elif args.data_option == 1:
        train_data = set_dataset(args.train_image_path, os.path.join(args.train_image_path, 'train.csv'), feature_extract_option=1, feature_path='/mit_resnet_train.pickle')
        val_data = set_dataset(args.val_image_path, os.path.join(args.val_image_path, 'validation.csv'), feature_extract_option=1, feature_path='/mit_resnet_val.pickle')

    if args.data_save_option == 0:
        with open(args.train_image_path + '/mit_resnet_train.pickle', 'wb') as output:
            pickle.dump(train_data.video_features, output, pickle.HIGHEST_PROTOCOL)

        with open(args.val_image_path + '/mit_resnet_val.pickle', 'wb') as output:
            pickle.dump(val_data.video_features, output, pickle.HIGHEST_PROTOCOL)

    train_x = train_data.video_features
    train_y = train_data.risk_one_hot
    train_y = np.delete(train_y, -1, axis=0)

    val_x = val_data.video_features
    val_y = val_data.risk_one_hot
    val_y = np.delete(val_y, -1, axis=0)

    model = Models(nb_epoch=args.nb_epoch, batch_size=args.batch_size)
    model.build_transfer_LSTM_model3(input_shape=train_x.shape[1:])
    model.class_weights = {0: 1, 1: 1}
    model.train_model(train_x, train_y, val_x, val_y, print_option=1)

    print('   safe | dangerous  \n', model.model.predict_proba(val_x[29:30]))
    model.plot_auc()
#    model.model.save(args.model_name)


