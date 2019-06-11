# DeepTL-Lane-Change-Classification
Infers the risk level of lane change video clips. Utilizes transfer learning (TL). The related paper is submitted to IEEE ITSC 2019. 


## Features: 
* A novel deep spatiotemporal framework for risky action detection in short lane change video clips
* using solely a camera for the task
* Two versions available with trained weights! : ResNet50 (TL) + LSTM and MaskRCNN (TL) + CNN + LSTM.
* Two lane change video samples are provided in data/input/lc_samples_50frames.tar.gz 


## Examples:

Here are two examples. The left lane change is classified as safe and the right lane change is classifed as risky.


<img src="example_gifs/260.gif" title="Safe Lane Change Example" width="400" hspace="10"> <img src="example_gifs/697.gif" title="Risk Lane Change Example" width="400" hspace="10"> 

## Installation:


## Prediction example:

1- Extracts the contents of lc_samples_50frames.tar.gz in data/input. The folder structure after extraction should look like the following:

    .
    ├── data
    │   ├── input 
    │   │   ├── 1                                    # The first lane change sequence
    │   │   │   ├── 260_7308.jpg                     # First frame of LC1
    │   │   │   ├── 260_7317.jpg                     # Second frame of LC1
    │   │   │   ├── ...      
    │   │   │   ├── 260_7317.jpg                     # 50th frame of LC1. The original footage was captured with 30 fps. However, we can only share the downsampled version here :(. Each lane chagne sequecne was downsampled to have 50 frames in total.
    │   │   ├── 2                                    # The second lane change sequence
    │   │   │   ├── 697_14899.jpg                    # First frame of LC2
    │   │   │   ├── ...                
    │   ├── extracted_features                       # 
    ├── example_gifs                    
    ├── lane_change_risk_detection                   # 
    ├── resnet_lstm.h5                               # The trained model. Weights and architecture 
    ├── sample_prediction_resnet_lstm.py             # This script infers the risk level of the lane changes in data/input/..
    ├── ...
 
2- Run sample_prediction_resnet_lstm.py:

This will infer the risk level of all lane changes in the folder  data/input/.. and print the result. For the provided sample lane change sequences the result should look like

      
      safe | dangerous 
      [[0.9169345  0.08306556]
      [0.13948527 0.8605147 ]]

This means the first lane change is classified as safe (0.91) and the second lane change is classified as dangerous (0.86).

3- If you want to use the trained model to infer the risk level of your own lane change data, simply add more lane change folders under data/input/.. with numerical foldernames such as 3, 4, .... The sample_prediction_resnet_lstm.py will infer the risk level of all lane changes in the input folder.


## Credits:

[Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN) is utilized for the segmentation part in this project.

## Coming soon:

-MaskRCNN segmentation mask transfer!

-The full release will be available soon. Stay tuned for updates!
