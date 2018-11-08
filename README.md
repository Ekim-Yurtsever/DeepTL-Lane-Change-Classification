# DeepTL-Lane-Change-Classification

1 - Extract features from image frames with pre-trained very deep network 

2 - Use the features as inputs of an LSTM network

3 - Train the LSTM with the target labels (risk scores)

4 - Compare the results with different pre-trained very deep networks

5 - Compare the results with different video classification methods:

     i -  CNN frame by frame
     
     ii - CNN + LSTM trained together
     
     iii - 3d CNN
     
     iv - transfer learning + simple MLP
