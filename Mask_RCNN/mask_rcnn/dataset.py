import os
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tqdm import tqdm


class DataSet:

    def __init__(self):

        self.features = []
        self.model = []

    def extract_feature(self, img_path):
        img = image.load_img(img_path, target_size=(331, 331))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        self.features = self.model.predict(x)

    def save(self, save_path='test.txt'):
        np.savetxt(save_path, self.features, delimiter=',')

    def load(self, load_path='test.txt'):
        self.features = np.loadtxt(load_path, delimiter=',')

    def iterate_folder(self, main_foldername='test_images', save_main_foldername='extracted_features'):
        foldernames = [f for f in os.listdir(main_foldername) if
                       f.isnumeric() and not f.startswith('.')]

        for foldername in tqdm(foldernames):
            filenames = sorted(os.listdir(main_foldername + '/' +foldername))

            for filename in filenames:
                img_path = main_foldername + '/' + foldername + '/' + filename
                self.extract_feature(img_path)
                if not os.path.exists(save_main_foldername + '/' + foldername):
                    os.makedirs(save_main_foldername + '/' + foldername)
                self.save(save_path=save_main_foldername + '/' + foldername + '/' + filename[:-4]+'.txt')
