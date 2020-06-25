# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
#from pyimagesearch.smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


#Model for extracting features at the sub-region level
class CNN_sub_region:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

# CONV => POOL layer set, a kernel size of 5 × 5, stride step of 2
		model.add(Conv2D(32, (5, 5),strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("tanh"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1)))
		model.add(Dropout(0.25))

# CONV => POOL layer set, a kernel size of 5 × 5, stride step of 2
		model.add(Conv2D(64, (5, 5),strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("tanh"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1)))
		model.add(Dropout(0.25))
        
# CONV => POOL layer set, a kernel size of 5 × 5, stride step of 2
		model.add(Conv2D(256, (5, 5),strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("tanh"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1)))
		model.add(Dropout(0.25))

# CONV => POOL layer set, a kernel size of 5 × 5, stride step of 2
		model.add(Conv2D(1024, (5, 5),strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("tanh"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Flatten())
		model.add(Dense(4096))
		model.add(Activation("tanh"))
		model.add(BatchNormalization())
		#model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model

#Model for extracting features at the character level
class CNN_character:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

# CONV => POOL layer set, a kernel size of 5 × 5, stride step of 2
		model.add(Conv2D(32, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("tanh"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

# CONV => POOL layer set, a kernel size of 5 × 5, stride step of 2
		model.add(Conv2D(64, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("tanh"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
        
# CONV => POOL layer set, a kernel size of 5 × 5, stride step of 2
		model.add(Conv2D(256, (5, 5),strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("tanh"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

# Fully connected layer
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("tanh"))


		model.add(BatchNormalization())
		#model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))


		# return the constructed network architecture
		return model

#Parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-m2", "--model2", required=True,
	help="path to output trained model2")
ap.add_argument("-p2", "--plot2", required=True,
	help="path to output accuracy/loss plot")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())

print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of SmallVGGNet), and store the image in the
	# data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)
	# extract the class label from the image path and update the
	# labels list
	#label = imagePath.split(os.path.sep)[-2]
	label = imagePath.split(os.path.sep)[-3]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#Train Sub level network
modelSubLvl = CNN_sub_region.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

#Train character level network
modelCharLvl = CNN_character.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

#Parameters for training
INIT_LR = 0.01
EPOCHS = 50
BS = 32


print("[INFO] training subLvL network...")
modelSubLvl.compile(loss="categorical_crossentropy", optimizer="nadam",
	metrics=["accuracy"])
H = modelSubLvl.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=BS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = modelSubLvl.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SubLvl")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
modelSubLvl.save(args["model"])
# = open(args["label_bin"], "wb")
#f.write(pickle.dumps(lb))
#f.close()



# train character level network
modelCharLvl.compile(loss="categorical_crossentropy", optimizer="nadam",
	metrics=["accuracy"])
# train the network
H = modelCharLvl.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=BS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = modelCharLvl.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (CharLvL)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot2"])
# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
modelCharLvl.save(args["model2"])

f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
#f.close()



'''
-d characters_for_style_classification/ -m Models/model3.h5 -m2 Models/model4.h5 -p Plots/plot3 -p2 Plots/plot4 - l mlb.pickle
'''