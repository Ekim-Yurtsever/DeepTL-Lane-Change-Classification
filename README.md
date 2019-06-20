# DeepTL-Lane-Change-Classification
Infers the risk level of lane change video clips. Utilizes transfer learning (TL). The related paper is submitted to IEEE ITSC 2019. The preprint of the article is available at https://arxiv.org/abs/1906.02859 


## Features: 
* A novel deep spatiotemporal framework for risky action detection in short lane change video clips
* using solely a camera for the task
* Two versions are available with trained weights! : ResNet50 (TL) + LSTM and MaskRCNN (TL) + CNN + LSTM.
* Two lane change video samples are provided in data/input/lc_samples_50frames.tar.gz 


## Examples:

Here are two examples. The left lane change is classified as safe and the right lane change is classifed as risky.


<img src="example_gifs/260.gif" title="Safe Lane Change Example" width="400" hspace="10"> <img src="example_gifs/697.gif" title="Risk Lane Change Example" width="400" hspace="10"> 

## Installation:
    $git clone https://github.com/Ekim-Yurtsever/DeepTL-Lane-Change-Classification.git
    $cd DeepTL-Lane-Change-Classification/
    $pip install -r requirements.txt

## Prediction example:

1- Extract the contents of lc_samples_50frames.tar.gz in data/input. The folder structure after extraction should look like the following:

    .
    ├── Mask_RCNN                                    # Mask RCNN implementation by Matterport
    ├── data                                         # Data dir
    │   ├── input 
    │   │   ├── 1                                    # The first lane change sequence
    │   │   │   ├── 260_7308.jpg                     # First frame of LC1
    │   │   │   ├── 260_7317.jpg                     # Second frame of LC1
    │   │   │   ├── ...      
    │   │   │   ├── 260_7317.jpg                     # 50th frame of LC1. The original footage was captured with 30 fps. However, we can only share the downsampled version here :(. Each lane change sequence was downsampled to have 50 frames in total.
    │   │   ├── 2                                    # The second lane change sequence
    │   │   │   ├── 697_14899.jpg                    # First frame of LC2
    │   │   │   ├── ...                
    │   ├── extracted_features                       
    │   ├── masked_images                      
    ├── example_gifs                    
    ├── lane_change_risk_detection                   # Source code of this project
    ├── test                                         # test prediction and training scripts to get started!
    ├── ...
 
2- Run test/sample_prediction_resnet_lstm.py:

This will infer the risk level of all lane changes in the folder  data/input/.. and print the result. This script uses the ResNET backbone for risk inference. This is faster than MaskRCNN based semantic mask transfer backbone, but has lower performance. For the provided sample lane change sequences the result should look like the following:

      
      safe | dangerous 
      [[0.9169345  0.08306556]
      [0.13948527 0.8605147 ]]

This means the first lane change is classified as safe (0.91) and the second lane change is classified as dangerous (0.86).

PLEASE NOTE: If you use different versions of keras or tensorflow-GPU, the trained models will either not work or give false results!! The trained models work only with the specific tensorflow-gpu version which I used to train with our data (the data is not open access)! I will add cpu models later. IF YOU DONT GET THE RESULT MENTIONED ABOVE PLEASE CHECK DEPENDINCIES. Here are the requirments:

### Requirments:
 
    absl-py==0.7.1
    astor==0.8.0
    gast==0.2.2
    google-pasta==0.1.7
    grpcio==1.21.1
    h5py==2.9.0
    Keras==2.2.2
    Keras-Applications==1.0.4
    Keras-Preprocessing==1.0.2
    Markdown==3.1.1
    numpy==1.14.5
    opencv-python==4.1.0.25
    pandas==0.24.2
    Pillow==6.0.0
    pkg-resources==0.0.0
    protobuf==3.8.0
    python-dateutil==2.8.0
    pytz==2019.1
    PyYAML==5.1.1
    scipy==1.3.0
    six==1.12.0
    tensorboard==1.10.0
    tensorflow-estimator==1.14.0
    tensorflow-gpu==1.10.1
    termcolor==1.1.0
    tqdm==4.32.2
    Werkzeug==0.15.4
    wrapt==1.11.2


3- Run test/sample_prediction_maskRCNN_lstm.py:

This will do the same thing with the MaskRCNN backbone. Also, the masked images will be saved to /data/masked_images . This method has higher performance but it is slower. The risk inference result should look like the following:

    safe | dangerous 
    [[0.94456106 0.05543892]
    [0.07597142 0.92402864]]

4- If you want to use the trained model to infer the risk level of your own lane change data, simply add more lane change folders under data/input/.. with numerical foldernames such as 3, 4, .... The sample_prediction_resnet_lstm.py will infer the risk level of all lane changes in the input folder.

## Architecture
The architecture of the model is shown below. SMT stands for "Semantic Mask Transfer." Please check our [paper](https://arxiv.org/abs/1906.02859) for details.

<img src="example_gifs/architectures2.png" title="Model architecture"> 

## Credits:

[Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN) is utilized for the segmentation part in this project.

## Coming soon:

-The full release will be available soon. Stay tuned for updates!
