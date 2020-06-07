from numpy import mean
from numpy import std

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from imutils import paths
from keras.utils import to_categorical

import argparse
import cv2
import random
import os
import numpy as np

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[0], trainX.shape[1], trainy.shape[0]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(trainX, trainY, testX, testY, repeats=10):
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainY, testX,testY)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output trained model")
# ap.add_argument("-l", "--label-bin", required=True,
# 	help="path to output label binarizer")
# ap.add_argument("-p", "--plot", required=True,
# 	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # Read the image
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    # Compute moments
    moments = cv2.moments(image)

    # Compute Hu moments
    hu_moments = cv2.HuMoments(moments).flatten()

    # Store data
    data.append(hu_moments)

    # Store labels
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Convert data to numpy arrays
data = np.array(data, dtype="float")
data = np.expand_dims(data, axis=2)
labels = np.array(labels)
print(type(data))
print(data.shape)
print(data.shape[0])
print(data.shape[1])


# Split data into train/test
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

run_experiment(trainX, trainY, testX, testY)