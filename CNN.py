# Author: Randa Yoga Saputra
# Version: 1
# Purpose: Class of pretrained models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class ConfusionMatrix:
    """Instantiate a new data structure, a confusion matrix, which can be used to 
    assess the accuracy of classifications.  Particularly useful for multiclass
    classification because it elucidates the specific classification error that 
    occurred.  This datastructure is built on top of an np.array.
    """
    def __init__(self, class_names: list, nameLengthLimit: int = 4):
        self.class_names = class_names
        self.nameLengthLimit = nameLengthLimit
        self.ERR = {i if len(i) <= nameLengthLimit else i[:4]:{"total":0, "actual":0} for i in class_names}
        self.matrix = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype='object')
        for i, label in enumerate(class_names):
            self.matrix[0, i + 1] = label[:nameLengthLimit]
            self.matrix[i+1, 0] = label[:nameLengthLimit]
        self.pad_matrix()
        self.set_top_left_brick()

    def __repr__(self):
        return f"{self.matrix}"

    def set_top_left_brick(self, char: str = "x"):
        """set upper left corner of matrix to a unique set of characters.
        for design pupose only (non essential).
        """
        self.matrix[0,0] = "".join([char for i in list(self.matrix[0,0])])
    
    def max_length(self) -> int:
        """Find element with max length in matrix.
        """
        return max([len(str(i)) for i in self.matrix.flatten()])

    def pad_cell(self, idx, char=" "):
        """ pad a single matrix element with:
        len(element with max length) - len(ith element) spaces (chars).
        """
        self.matrix[idx] = str(self.matrix[idx]).strip()
        self.matrix[idx] = str(self.matrix[idx]) + char * (self.max_length() - len(str(self.matrix[idx])))

    def pad_matrix(self, char: str = " "):
        """ pad matrix elements with:
        len(element with max length) - len(ith element) spaces.
        """
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                self.pad_cell((i,j), char=char)

    def increment(self, label: str, key: str):
        self.ERR[label[:self.nameLengthLimit]][key] += 1

    def increment_matrix(self, key: str, value: str):
        idx = (self.class_names.index(key) + 1, self.class_names.index(value) + 1)
        self.matrix[idx] = int(self.matrix[idx])+1
        self.pad_cell(idx)

class ReferenceModel:
    """This class will serve as a reference to compare my models too.
    simple predictions strategies will be defined here.  The CNN's prediction
    accuracy should perform better than these methods
    """
    def __init__(self, classes: list):
        self.classes = classes
    def random_predict(self):
        return np.random.choice(self.classes)

class Residual(tf.keras.Model):
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

class ResnetBlock(tf.keras.layers.Layer):
    """Used in creating a ResNet CNN.
    """
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        # Iterate directly over self.residual_layers (which is a list)
        for layer in self.residual_layers:
            X = layer(X)
        return X

class CNN:
    """An organized collection of various convolutional neural network architectures.
    """
    def __init__(self, img_size, channels, classes):
        self.model = None
        self.channels = channels
        self.img_height, self.img_width = img_size
        self.classes = classes

    def get_num_classes(self):
        """Return number of classes used in this network.
        """
        return len(self.classes)
    def cnn_tf(self):
        """CNN Architecture defined in tensorflow tutorial
        """
        self.model = Sequential([
			#layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_height, self.img_width, self.channels)),
			layers.Conv2D(16, 3, padding='same', activation='relu'),
			layers.MaxPooling2D(),
			layers.Conv2D(32, 3, padding='same', activation='relu'),
			layers.MaxPooling2D(),
			layers.Conv2D(64, 3, padding='same', activation='relu'),
			layers.MaxPooling2D(),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(self.get_num_classes())
        ])
    def LeNet(self):
	    self.model = keras.models.Sequential([
			layers.experimental.preprocessing.Rescaling(1./255, input_shape=(100, 100, self.channels)),
			keras.layers.Conv2D(16, kernel_size=5, strides=1, input_shape=(self.img_height, self.img_width, self.channels),  activation='tanh', padding='same'),
			keras.layers.AveragePooling2D(), 
			keras.layers.Conv2D(32, kernel_size=5, strides=1, activation='tanh', padding='valid'),
			keras.layers.AveragePooling2D(), 
			keras.layers.Flatten(), 
			keras.layers.Dense(120, activation='tanh'), 
			keras.layers.Dense(84, activation='tanh'), 
			keras.layers.Dense(4, activation='softmax') #Output layer
		])
    def AlexNet(self):
        # 3. AlexNet architecture:
		# proof of concept that deep learning works.
		# changed filters from 200 to 96
		# why are there no parameters in pool layer? -> parameters are learned weights (conv layers)
		# pooling, use rule to look at pooling region, take max, nothing learned.
		# deep learning.
        self.model=keras.models.Sequential([
			layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_height, self.img_width, self.channels)), # added rescaling layer
			# conv2d (Conv2D) (None, 23, 23, 200)       72800 | calculated by (input shape - filterSize)//stride + 1
			keras.layers.Conv2D(filters=200, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(self.img_height,self.img_width, self.channels)),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPool2D(pool_size=(2,2)),
			keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPool2D(pool_size=(3,3)),
			keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPool2D(pool_size=(2,2)),
			keras.layers.Flatten(),
			keras.layers.Dense(1024,activation='relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(1024,activation='relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(self.get_num_classes(),activation='softmax')  # set final dense layer to map to # classes.
		])
    def vgg(self, conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
        """the function produces by defualt the vgg-11 cnn architecture,
        but a different conv_architecture argument can be provided.
        """
        # massive improvement from AlexNet (in theory)
		# uses smaller filters, but why?
		# similar to AlexNet in that it consists of two principle parts: convolution/pooling part & fully
		# connected part.
		# implementation: http://d2l.ai/chapter_convolutional-modern/vgg.html
		# instead of considering invidual neurons we will think of blocks of repeating layers.
        def vgg_block(num_convs, num_channels):
            blk = tf.keras.models.Sequential()
            for _ in range(num_convs):
                blk.add(
                    tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                        padding='same', activation='relu'))
            blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            return blk
        net = tf.keras.models.Sequential()
        # The convulational part
        for (num_convs, num_channels) in conv_arch:
            net.add(vgg_block(num_convs, num_channels))
        # The fully-connected part
        net.add(
            tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.get_num_classes())]))
        self.model = net

    def ResNet(self):
        """Implements the ResNet CNN architecture as defined in D2L.
        At the heart of this architecture is the idea that every additional
        laye should easily contain the identity function: f(x) = x.
        """
        self.model = tf.keras.Sequential([
        # The following layers are the same as b1 that we created earlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # The following layers are the same as b2, b3, b4, and b5 that we
        # created earlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=self.get_num_classes())])

    def compile(self, learn_rate=0.001):
        """If learn rate is not a float use Adam optimizer.
        """
        self.model.compile(
		optimizer = tf.optimizers.SGD(learning_rate=learn_rate) if type(learn_rate) == type(0.1) else 'adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

