
from keras.models import load_model
from lane_change_risk_detection.dataset import DataSet
from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects
import os

dir_name = os.path.dirname(__file__)
dir_name = os.path.dirname(dir_name)

image_path = os.path.join(dir_name, 'data/input/')
masked_image_path =os.path.join(dir_name, 'data/masked_images/')

masked_image_extraction = DetectObjects(image_path, masked_image_path)
masked_image_extraction.save_masked_images()

data = DataSet()
data.read_video(masked_image_path, option='fixed frame amount', number_of_frames=10, scaling='scale', scale_x=0.1, scale_y=0.1)

model = load_model('maskRCNN_CNN_lstm.h5')
print(' safe | dangerous \n', model.predict_proba(data.video))