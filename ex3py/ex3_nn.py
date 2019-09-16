## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

#pillow library
from PIL import Image

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

import predict

from PIL import Image


## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')


#  training data stored in arrays X, y
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in
# MATLAB where there is no index 0
y[y == 10] = 0

# get number of examples in dataset
m = y.size

# randomly permute examples, to be used for visualizing one
# picture at a time
indices = np.random.permutation(m)

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)

#sel = sel(1:1);

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the .mat file, which returns a dictionary
weights = loadmat('ex3weights.mat')

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict.project(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}#'.format(np.mean(pred == y) * 100))


#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pred = predict.project(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')

## ================= Part 4: Fun Predict =================
# Draw a number yourself and predict that number
# start > paint > Resize > Pixel Radio button > Horizonta/Vertical 200
# fill square --> 'Fill with color' with Gray-50#
# Select color --> White; Brushes --> First Brush; Size -->  8px
# Draw a number
# Resize > Percentage ratdo button > Horizontal/Vertical 10 #
# This will give you 20x20 pixels image
# File > Save As > Save as type --> Monochrome Bitmap (*.bmp,dib)
# Type a filename > 'Ok' message
# Not working well yet for 0, 6, and 9
# Scope to improve (not the code, but handwriting :)

#img = Image.open('whoami.bmp')
#img = Image.open('knockknock.bmp')

#img = Image.open('uno.bmp')
#img = Image.open('tango.bmp')
#img = Image.open('musketeers.bmp')
#img = Image.open('quads.bmp')
#img = Image.open('five.bmp')
#img = Image.open('continents.bmp')

img = Image.open('wonders.bmp')

img.load()
data = np.asarray( img, dtype="double" )


data = data.T
#A = np.squeeze(np.asarray(data))
A = np.array(data).flatten()

mu = np.mean(A, axis = 0)
sigma = np.std(A, ddof=1, axis = 0)
A_norm = (A - mu) / sigma

utils.displayData(A_norm)

pred = predict.project(Theta1, Theta2, A_norm)
print('\nNeural Network FUN Prediction: digit', ( pred % 10))
