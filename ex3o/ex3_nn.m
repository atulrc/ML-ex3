%% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:1);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Pameters ================
% In this part of the exercise, we load some pre-initialized
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat');

%% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:2
    % Display
    fprintf('\nDisplaying Example Image\n');
%    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));

    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end

%% ================= Part 4: Fun Predict =================
% Draw a number yourself and predict that number
% start > paint > Resize > Pixel Radio button > Horizonta/Vertical 200
% fill square --> 'Fill with color' with Gray-50%
% Select color --> White; Brushes --> First Brush; Size -->  8px
% Draw a number
% Resize > Percentage ratdo button > Horizontal/Vertical 10 %
% This will give you 20x20 pixels image
% File > Save As > Save as type --> Monochrome Bitmap (*.bmp,dib)
% Type a filename > 'Ok' message
% Not working well yet for 0, 6, and 9
% Scope to improve (not the code, but handwriting :)

%a = imread('whoami.bmp');
%a= imread('knockknock.bmp');

%a = imread('uno.bmp');
%a = imread('tango.bmp');
%a = imread('musketeers.bmp');
%a = imread('quads.bmp');
%a = imread('five.bmp');
%a = imread('continents.bmp');
a = imread('wonders.bmp');   % Including you

A = (a(:))';

B = cast(A,"double");

B = (B - mean(B)) /(std(B));
displayData(B);

pred = predict(Theta1, Theta2, B);
fprintf('\nNeural Network FUN Prediction: %d (digit %d)\n', pred, mod(pred, 10));
