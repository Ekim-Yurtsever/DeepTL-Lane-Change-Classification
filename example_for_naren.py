from dataset import DataSet
from keras.optimizers import Adam
from models import Models


data = DataSet()
data.read_features(feature_size=2048, feature_path='/media/ekim-hpc/hdd1/lane_change_risk_detection/extracted_features/res_net_50_imagenet', number_of_frames=20)
data.read_risk_data("LCTable.csv")
data.convert_risk_to_one_hot(risk_threshold=0.05)
#data = DataSet.loader("/media/ekim-hpc/hdd1/lane_change_risk_detection/saved data/dataset_resnet_features_5percent.pickle")
filename = "resnet_f_03"

timesteps = data.video_features.shape[1]
nb_samples = data.video_features.shape[0]
nb_features = data.video_features.shape[2]

class_weight = {0: 0.05, 1: 0.95}
training_to_all_data_ratio = 0.9
nb_epoch = 100
batch_size = 32
optimizer = Adam(lr=1e-4, decay=1e-2)

input_data = data.video_features
target_data = data.risk_one_hot

models = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
models.build_transfer_LSTM_model(input_shape=input_data.shape[1:], optimizer=optimizer)
X_train, y_train, X_test, y_test = models.split_training_data(input_data, target_data, training_to_all_data_ratio)

models.train_model(X_train, y_train, X_test, y_test)

models.plot_auc()



