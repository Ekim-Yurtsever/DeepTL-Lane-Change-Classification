from dataset import *
import numpy as np
np.random.seed(1337)
from keras.optimizers import SGD, Adam
from models import Models
import os

data_path = "/media/ekim-hpc/hdd1/lane_change_risk_detection/saved data/"
filenames = ['dataset_InceptionResNetV2_5percent.pickle', 'dataset_resnet_features_5percent.pickle',
             'dataset_NASNetLarge_5percent.pickle', 'dataset_mobilenet_v2.pickle']
data = DataSet.loader(data_path+filenames[1])
filename = "resnet_f_03"

timesteps = data.video_features.shape[1]
nb_samples = data.video_features.shape[0]
nb_features = data.video_features.shape[2]

class_weight = {0: 0.05, 1: 0.95}
training_to_all_data_ratio = 0.9
nb_cross_val = 5
nb_epoch = 1000
batch_size = 32
optimizer = Adam(lr=1e-4, decay=1e-2)

Data = data.video_features
label = data.risk_one_hot

models = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
models.build_transfer_LSTM_model(input_shape=Data.shape[1:], optimizer=optimizer)
models.train_n_fold_cross_val(Data, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val,
                              print_option=1, plot_option=1, epoch_resolution=int(nb_epoch*0.1), save_option=1,
                              save_path="result"
                                        "/"+filename+".png")

print('Average tenfold cross validation loss:')
print(sum(models.m_fold_cross_val_results)/len(models.m_fold_cross_val_results))

if os.path.isfile("results/"+filename+"_average_auc.txt"):
    file = open("results/" + filename + "_average_auc.txt", "a")
    file.write(str(sum(models.m_fold_cross_val_results) / len(models.m_fold_cross_val_results))+"\n")
    models.model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.close()
else:
    file = open("results/"+filename+"_average_auc.txt", "w")
    file.write(str(sum(models.m_fold_cross_val_results)/len(models.m_fold_cross_val_results))+'\n')
    models.model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.close()


