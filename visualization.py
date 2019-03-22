import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications import ResNet50, MobileNetV2, NASNetMobile, NASNetLarge, DenseNet201, InceptionResNetV2, VGG19, Xception
from keras.models import Model
from matplotlib import pyplot as plt
import os


# With the trained model do the following:
# 1- Given a single image and intermediate CON layer, create an activation image of that CONV layer
# 2- Given a folder and intermediate CONV layer, create activation images for every image in the folder
# --------------------------------------
# 3- Given an intermediate CONV layer id, create filter weight images
# 4- Given an intermediate CONV layer id, generate maximal activation inputs for each filter
# --------------------------------------
# 5-Given an image and intermediate CONV layer id, extrapolate the activation of a CONV filter over the image and plot
# (i.e plot attention)

# https://github.com/raghakot/keras-vis
# Check this^ repo for:
# Activation maximization
# Saliency maps
# Class activation maps

class Visualization:

    def __init__(self):
        self.feature = []
        self.model = []
        self.vis_layer_id = []
        self.activation_model = []

    def visualize_activation_with_image(self, image_path, filter_id=0, layer_name='activation_49', save_option=0, save_path='default'):
        self.activation_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        activation_output = self.activation_model.predict(x)
        ax = plt.subplot(111)
        ax.axis('off')

        if save_option == 1:
            plt.imsave(save_path, activation_output[0, :, :, filter_id])
        elif save_option == 0:
            ax.imshow(activation_output[0, :, :, filter_id])

    def visualize_filters(self):
        # Visualizes CNN filters. i.e the weights of layers
        # todo
        self.feature = []

        # filter vis
        # 1- Visualize Maximal activation
        # 2- Visualize Filter weights
        # top_layer = model.layers[2]
        # plt.imshow(top_layer.get_weights()[0][:, :, :, 0].squeeze(), cmap='viridis')

    def visualize_features(self):
        # visualizes the output of CNN layers. (filter x input image)
        # todo
        self.feature = []

    def iterate_folder_function_wrapper(self, INPUT_DIR, OUTPUT_DIR, filter_id=0, layer_name='activation_49'):  #convert this to a wrapper
        # todo #convert this to a wrapper

        foldernames = [f for f in os.listdir(INPUT_DIR) if f.isnumeric() and not f.startswith('.')]
        foldernames.sort()
        for foldername in foldernames:
            CURRENT_INPUT_DIR = INPUT_DIR + foldername + '/'
            CURRENT_OUTPUT_DIR = OUTPUT_DIR + foldername + '/'
            if not os.path.isdir(CURRENT_OUTPUT_DIR):
                os.mkdir(CURRENT_OUTPUT_DIR)
            filenames = sorted(os.listdir(CURRENT_INPUT_DIR))
            for filename in filenames:
                print(filename)
                if not filename.startswith('.'):
                    #kwargs = kwargs + current_input + current_output
                    self.visualize_activation_with_image(CURRENT_INPUT_DIR + filename, filter_id=filter_id, layer_name=layer_name,
                                                         save_option=1, save_path=CURRENT_OUTPUT_DIR + filename)
                    #function(**kwargs)


# img_path = '/Users/ekimmac/Dropbox/PhD/projects/maskRCNN_detection/test_images/1/1_1096.jpg'
# # img_path = '/Users/ekimmac/Dropbox/PhD/projects/maskRCNN_detection/test_images/474/474_11128.jpg'
#
#
# base_model = ResNet50(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('activation_49').output)
# #1st conv = 'conv1'
# #res4f_branch2c
# # 'res3d_branch2c'
# # activation_49
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# activation_output = model.predict(x)
# plt.matshow(activation_output[0, :, :, 0], cmap='viridis')

# -------------------------------------------------------
img_path = '/Users/ekimmac/Dropbox/PhD/projects/maskRCNN_detection/test_images/1/1_1096.jpg'
visualizer = Visualization()
visualizer.model = ResNet50(weights='imagenet')





INPUT_DIR = '/Users/ekimmac/Dropbox/PhD/projects/maskRCNN_detection/test_images/1/'
OUTPUT_DIR = '/Users/ekimmac/Dropbox/PhD/projects/git_projects/lane_change_classification/DeepTL-Lane-Change-Classification/processed_images/'

visualizer.iterate_folder_function_wrapper(INPUT_DIR=INPUT_DIR, OUTPUT_DIR=OUTPUT_DIR, filter_id=0, layer_name='activation_49')

visualizer.visualize_activation_with_image(img_path, filter_id=0, layer_name='conv1')
# 1st conv = 'conv1'
# res4f_branch2c
# 'res3d_branch2c'
# activation_49
