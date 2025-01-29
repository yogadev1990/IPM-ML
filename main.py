# Author: Randa Yoga Saputra
# Version: 4
# Purpose: Model training based on arch
# How to start: python main.py --build_model --overwrite

import numpy as np
import os
import argparse
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# Custom modules imported from other python files in this directory
# ------------------------------------------------------------------
# my module (CNN) - convolutional neural network architectures.
from CNN import CNN, ReferenceModel, ConfusionMatrix
# helper functions used for data manipulation
from nutils import findTrueKey
# added to fix type annotations
from typing import Tuple

# 0) Globals
# Create an argument parser
argParser = argparse.ArgumentParser(
	description="Machine learning oral lesion classifier.",
    epilog=u"\U0001F432"
)
argParser.add_argument('--build_model', action='store_true') 
argParser.add_argument('--overwrite', action='store_true') 

args = argParser.parse_args()

BUILD_MODEL = args.build_model # True create a new model, if false attempt to use the most "fresh" saved model
OVERWRITE = args.overwrite # overwrite freshest model (when building new model)
PLOT = True # show plots
TRAIN_PLOT = True # show training results

LOAD_INDEX = -1 # Automatically choose freshest model when BUILD_MODEL = False
path = "../Oral Lesion Database"
img_width, img_height = 100, 100 # dimensions of image in pixels

# Choosing a CNN architecture.  Please set one of the following values to True in the arch dictionary.
arch = {
	"cnn_tf": False,
	"LeNet": False,
	"AlexNet": False,
	"VGG": False,
	"ResNet": True
}
arch_key = findTrueKey(arch) # given a dictionary, return the key with the True value set.

# various classification schemes can occur with this program.  YOu can choose which scheme to use
# by setting a specific modelType to True.  there should only be one True value.
# model type - lesion v. non lesion/ herpes v. ssc/ lip v. tongue/ all
modelType = {
	"lesion_v_nonlesion": False, # binary classification: lesion versus not a lesion
	"ganas_v_jinak": True, # binary classification: ganas versus jinak
	"lip_v_tongue": False, # binary classification: lip tissue versus tongue tissue
	"herpes_v_scc": False, # binary classification: herpes tissue versus squamous cell carcinoma tissue
	"all": False # quaternary classification: herpes versus squamous cell carcinoma versus lip versus tongue
}
dir_suffix = findTrueKey(modelType)

epochs = 100 # Number of epochs used when training specify here [] add to arg parse
learn_rate = 0.001 # used by CNN class when compiling model.  Set to non float value to use ADAM optimizer.
class_names = os.listdir(f"./data/traintest/traintest_{dir_suffix}")
reference = ReferenceModel(class_names) # non cnn - reference classifications.  Serve as control for experiments
model = None # place holder for machine learning model

def read_test_images(dim: Tuple[int, int], dir_suffix: str) -> Tuple[list, list, list]:
	"""
	read out of sample test images located in the 'predict' directory.
	input
		dim: tuple = (img_height: int, img_width: int)
		dir_suffix: str = name of directory holding the out of sample test images.
	output
		labels: list - Actual class labels for testing data. 
		images: list - Image data for testing data.
		names: list - Files names of images for testing data.
	"""
	path = f"./data/predict/predict_{dir_suffix}"
	labels, images, names = [], [], []
	for label in os.listdir(path):
		for img_path in os.listdir(os.path.join(path,label)):
			img = keras.preprocessing.image.load_img(
				os.path.join(path,label,img_path), target_size = dim
			)
			img_array = keras.preprocessing.image.img_to_array(img)
			img_array = tf.expand_dims(img_array, 0) # Create a batch
			labels.append(label)
			images.append(img_array)
			names.append(os.path.join(path,label,img_path))
	return (labels, images, names)

def main():
	if  BUILD_MODEL:
		print(f"Initiating {dir_suffix} classification.")

		# 2) Create train/ validation split for model development.
		# tf load
		batch_size = 23
		data_dir = f"./data/traintest/traintest_{dir_suffix}"

		# Use dataset from directory
		train_ds = tf.keras.preprocessing.image_dataset_from_directory(
			data_dir,
			validation_split=0.2,
			subset="training",
			seed=123,
			image_size=(img_height, img_width),
			batch_size=batch_size)
		val_ds = tf.keras.preprocessing.image_dataset_from_directory(
			data_dir,
			validation_split=0.2,
			subset="training",
			seed=123,
			image_size=(img_height, img_width),
			batch_size=batch_size)
		
		# Use data generator
		myPreciousSeed = 543
		train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
			validation_split=0.2,
			#rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)
		test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)#,rescale=1./255
		train_generator = train_datagen.flow_from_directory(
				data_dir,
				seed=myPreciousSeed,
				subset="training",
				target_size=(150, 150),
				batch_size=batch_size,
				class_mode='binary')
		validation_generator = test_datagen.flow_from_directory(
				data_dir,
				seed=myPreciousSeed,
				subset="validation",
				target_size=(150, 150),
				batch_size=batch_size,
				class_mode='binary')

		print(train_ds)
		print(f"Class names: {train_ds.class_names}, {len(train_ds.class_names)}")

		# 3) Configure for performance
		# buffered prefetching so you can yield data from disk
		# without blocking.  cache()/ prefecth()
		# Unknown error on this step!
		# AUTOTUNE = tf.data.AUTOTUNE
		# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
		# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

		# 4) Standardize the data	
		normalization = {
			"normal_1": False,
			"normal_2": False,
		}
		if normalization["normal_1"]:
			# Normal 1:
			# Scale between [-1, 1]
			normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5)
			normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x - 127.5), y))
		elif normalization["normal_2"]:
			# Normal 1:
			# Scale between [0, 1]
			normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
			normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
			normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
		else:
			# No normalization applied.
			pass

		# 5) Build the model
		# Network architectures:
		# initialize a convolutional neural network using the CNN class.
		network = CNN((img_height, img_width), 3, train_ds.class_names)
		if arch["cnn_tf"]:
			# 1. tensorflow tutorial architecture:
			arch_key = "cnn_tf"
			network.cnn_tf()
			network.compile(learn_rate="adam")
			print("created and compiled tensorflow tutorial architecture cnn.")
		elif arch["LeNet"]:
			# 2. LeNet-5 architecture:
			arch_key = "LeNet"
			network.LeNet()
			network.compile(learn_rate=learn_rate)
			print("created and compiled LeNet cnn.")
		elif arch["AlexNet"]:
			# 3. AlexNet architecture:
			arch_key = "AlexNet"
			network.AlexNet()
			network.compile(learn_rate=learn_rate)
			print("created and compiled AlexNet cnn.")
		elif arch["VGG"]:
			# 4. VGG-11 architecture:
			arch_key = "VGG"
			network.vgg()
			network.compile(learn_rate=learn_rate)
			print("created and compiled vgg cnn.")
		elif arch["ResNet"]:
			# 5. ResNet architecture:
			arch_key = "ResNet"
			network.ResNet()
			network.compile(learn_rate=learn_rate)
			print("created and compiled ResNet cnn.")
		else:
			raise Exception("No CNN architecture selected.  Please modify the arch dictionary.")

		model = network.model
		history = model.fit(
			train_ds,#train_ds, train_generator
			validation_data=val_ds,#val_ds, validation_generator
			epochs=epochs
		)
		model.summary()
		inc = 0 if OVERWRITE else 1 # if we don't want to create a new model, but rather update the freshest model.
		models_dir = sorted([int(i) for i in os.listdir(f"models/models_{dir_suffix}/{arch_key}")])
		idx = 0 if len(os.listdir(f"models/models_{dir_suffix}/{arch_key}")) == 0 else models_dir[-1] + inc
		
		model.save(f'models/models_{dir_suffix}/{arch_key}/{idx}/model_{idx}')
		print(f"\nmodel {idx} saved.")
		
		# 6) Visualize training results
		if TRAIN_PLOT:
			acc = history.history['accuracy']
			val_acc = history.history['val_accuracy']

			loss = history.history['loss']
			val_loss = history.history['val_loss']

			epochs_range = range(epochs)

			plt.figure(figsize=(8, 8))
			plt.subplot(1, 2, 1)
			plt.plot(epochs_range, acc, label='Training Accuracy')
			plt.plot(epochs_range, val_acc, label='Validation Accuracy')
			plt.legend(loc='lower right')
			plt.title('Training and Validation Accuracy')

			plt.subplot(1, 2, 2)
			plt.plot(epochs_range, loss, label='Training Loss')
			plt.plot(epochs_range, val_loss, label='Validation Loss')
			plt.legend(loc='upper right')
			plt.title('Training and Validation Loss')
			plt.show()
			
		# Visualize data
		if PLOT:
			plt.figure(figsize=(10, 10))
			for images, labels in train_ds.take(1):
				for i in range(9):
					ax = plt.subplot(3, 3, i + 1)
					plt.imshow(images[i].numpy().astype("uint8"))
					plt.title(class_names[labels[i]])
					plt.axis("off")
			plt.show()
	else:
		if len(os.listdir(f"models/models_{dir_suffix}/{arch_key}")) > 0:
			print("trying load model.")
			idx = os.listdir(f"models/models_{dir_suffix}/{arch_key}")[LOAD_INDEX]
			model = tf.keras.models.load_model(f'models/models_{dir_suffix}/{arch_key}/{idx}/model_{idx}')
			print("model",model)
		else:
			print(
				"No historical models found, please build a new model by setting BUILD_MODEL = True."
			)

	if model:
		print(f"Initiating {dir_suffix} classification.")
		print("model found.")
		# 7) predicting on new data:
		
		# antiquated out of sample error rate, use more modern confusion matrix below.
		eout_ref = {i:{"total":0, "actual":0} for i in class_names} # store reference eout
		eout = {i:{"total":0, "actual":0} for i in class_names} # store out of sample error per class

		# initialize 2 confusion matrices, 1 for the cnn model and 1 for the reference model.
		cmatrix = ConfusionMatrix(class_names, nameLengthLimit=6)
		cmatrix_ref = ConfusionMatrix(class_names, nameLengthLimit=6)

		labels_t, vectors_t, names_t = read_test_images((img_height, img_width), dir_suffix)
		print("initialized eout & test materials.")
		 
		for lt, vt, nt in zip(labels_t, vectors_t, names_t):
			# print("data: ",vt)
			predictions = model.predict(vt)
			print("predicted succesfully")
			score = tf.nn.softmax(predictions[0])

			model_predicted = class_names[np.argmax(score)]
			model_ref_predicted = reference.random_predict()

			print(
				f"\nCorrect: {lt}, Predicted: {model_predicted}, confidence: {100 * np.max(score):.2f}",
				f"\nImage name: {nt}"
			)

			# updating antiquated  eout dicts
			eout_ref[lt]["total"] += 1
			eout[lt]["total"] += 1

			# update model eout
			if lt == model_predicted:
				eout[lt]["actual"] += 1
			# update control eout
			if lt == model_ref_predicted:
				eout_ref[lt]["actual"] += 1
			
			# updating cost matrices
			cmatrix.increment_matrix(lt, model_predicted)
			cmatrix_ref.increment_matrix(lt, model_ref_predicted)
		
		# organize out of sample predictions into a simple console table.
		padding = max([len(i) for i in eout.keys()])
		print("\nOut of sample predictions\n----------------------------------")
		for lt in eout:
			print(
				f'CNN model:    {lt + " "*(padding - len(lt))} -> {eout[lt]["actual"]}/ {eout[lt]["total"]}',
				f'Control model:{lt + " "*(padding - len(lt))} -> {eout_ref[lt]["actual"]}/ {eout_ref[lt]["total"]}'
			)
		print()
		print("reference cost matrix: ")
		print(cmatrix_ref)
		print("model cost matrix: ")
		print(cmatrix)

if __name__ == "__main__": main()