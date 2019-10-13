# CustomLossFunction

## What is this repo for
This repo contains a custom loss function for machine learning models

## Loss function objective
Minimize the average euclidean distance between the K predictions and the target.

## Notes about the implementation
The function accepts tensors of two different shapes to match the requirements that was given, however inside the function reshaping and broadcasting of one of the tensors is carried out to match the shapes. While this function could be used as it is during training, another option would be to replicate the output vector K times to match the number of estimators and modify function to accept tensors of the same shape. Both solutions under the same starting point should give the same output.