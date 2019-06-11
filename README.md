# DeepTL-Lane-Change-Classification
Infers the risk level of lane change video clips. The related paper is submitted to IEEE ITSC 2019. 


## Features: 
* A novel deep spatiotemporal framework for risky action detection in short lane change video clips
* using solely a camera for the task
* Two versions available with trained weights! : ResNet50 + LSTM and MaskRCNN backbone + CNN + LSTM.


## Examples:

Here are two examples. The left lane change is classified as safe and the right lane change is classifed as risky.


<img src="example_gifs/260.gif" title="Safe Lane Change Example" width="400" hspace="10"> <img src="example_gifs/697.gif" title="Risk Lane Change Example" width="400" hspace="10"> 

## Installation:


## Prediction & training examples:


## Credits:

[Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN) is utilized for the segmentation part in this project.

## Coming soon:

The full release will be available soon. Stay tuned for updates!
