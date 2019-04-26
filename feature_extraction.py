from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet50, MobileNetV2, NASNetMobile, NASNetLarge, DenseNet201, InceptionResNetV2, VGG19, Xception
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from feature_dataset import DataSetFeature
from keras.preprocessing import image
import numpy as np
#import ssl

#ssl._create_default_https_context = ssl._create_unverified_context


base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten_1').output)
#
# base_model = MobileNetV2(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d_1').output)
#
# base_model = NASNetLarge(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d_1').output)
# base_model = InceptionResNetV2(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


feature_destination_path = '/media/ekim-hpc/hdd1/lane_change_risk_detection/extracted_features/test1'
image_path = '/media/ekim-hpc/hdd1/data/from Naren/risk_prediction/dataPreparation/lane_change_images'

dataset = DataSetFeature()
dataset.model = model
dataset.iterate_folder(main_foldername=image_path, save_main_foldername=feature_destination_path)


# img_path1 = '/Users/ekimmac/Dropbox/PhD/projects/maskRCNN_detection/test_images/1/1_1096.jpg'
# img_path2 = '/Users/ekimmac/Dropbox/PhD/projects/maskRCNN_detection/test_images/1/1_1296.jpg'
#
#
# def extract_feature(img_path):
#
#    img = image.load_img(img_path, target_size=(224, 224))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
#
#    return model.predict(x)
#
# feature1 = extract_feature(img_path1)
# feature2 = extract_feature(img_path2)
#
# np.savetxt('test.txt', feature1, delimiter=',')
# y = np.loadtxt('test.txt', delimiter=',')