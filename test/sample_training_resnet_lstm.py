from lane_change_risk_detection.dataset import *
from lane_change_risk_detection.models import Models
from lane_change_risk_detection.feature_dataset import DataSetFeature
from keras.applications import ResNet50
from keras.models import Model
import os

class_weight = {0: 0.05, 1: 0.95}
training_to_all_data_ratio = 0.9
nb_cross_val = 1
nb_epoch = 1000
batch_size = 32

dir_name = os.path.dirname(__file__)
dir_name = os.path.dirname(dir_name)

image_path = os.path.join(dir_name, 'data/input')
feature_destination_path =os.path.join(dir_name, 'data/extracted_features')

backbone_model = ResNet50(weights='imagenet')
backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.get_layer(index=-2).output)

feature_dataset = DataSetFeature()
feature_dataset.model = backbone_model
feature_dataset.iterate_folder(main_foldername=image_path, save_main_foldername=feature_destination_path)

data = DataSet()
data.read_features(feature_size=2048, feature_path=feature_destination_path, number_of_frames=50)
data.read_risk_data("LCTable.csv")
data.convert_risk_to_one_hot(risk_threshold=0.05)

Data = data.video_features
label = data.risk_one_hot

model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
model.build_transfer_LSTM_model(input_shape=Data.shape[1:])
model.train_n_fold_cross_val(Data, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
model.model.save('resnet_lstm.h5')
