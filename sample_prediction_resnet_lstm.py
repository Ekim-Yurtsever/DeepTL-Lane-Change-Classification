from keras.models import load_model, Model
from keras.applications import ResNet50
from lane_change_risk_detection.dataset import *
from lane_change_risk_detection.feature_dataset import DataSetFeature

dir_name = os.path.dirname(__file__)
image_path = os.path.join(dir_name, 'data/input')
feature_destination_path =os.path.join(dir_name, 'data/extracted_features')

backbone_model = ResNet50(weights='imagenet')
backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.get_layer(index=-2).output)

feature_dataset = DataSetFeature()
feature_dataset.model = backbone_model
feature_dataset.iterate_folder(main_foldername=image_path, save_main_foldername=feature_destination_path)

data = DataSet()
data.read_features(feature_size=2048, feature_path=feature_destination_path, number_of_frames=50)

model = load_model('resnet_lstm.h5')
print('  safe | dangerous \n', model.predict_proba(data.video_features))



