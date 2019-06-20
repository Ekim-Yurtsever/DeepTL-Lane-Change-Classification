from keras.models import load_model, Model
from keras.applications import ResNet50
from lane_change_risk_detection.dataset import DataSet
import os

dir_name = os.path.dirname(__file__)
dir_name = os.path.dirname(dir_name)

image_path = os.path.join(dir_name, 'data/input')
feature_destination_path =os.path.join(dir_name, 'data/extracted_features')

backbone_model = ResNet50(weights='imagenet')
backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.get_layer(index=-2).output)

data = DataSet()
data.model = backbone_model
data.extract_features(image_path, option='fixed frame amount', number_of_frames=50)

model = load_model('resnet_lstm.h5')
print('  safe | dangerous \n', model.predict_proba(data.video_features))



