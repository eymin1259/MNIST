# Implement a Handwritten Numbers classification using CNN for architecture
import numpy;
import scipy.special; # sigmoid function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ConvolutionNetwork :
    # building CNN
    def __init__(self) :
        # initialize CNN
        self.model = tf.keras.Sequential()
        # 1. first convolution layer
        # relu activation funtion -> to avoid negative numbers
        self.model.add(layers.Convolution2D(16, (3,3), input_shape=(28,28,1), activation='relu')) # numfilter, filterShape, inputShape
        # 2. Maxpooling
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        # 3. second convolution layer
        self.model.add(layers.Convolution2D(32, (3,3), activation='relu'))
        # 4. Maxpooling
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        # 5. third convolution layer
        self.model.add(layers.Convolution2D(64, (3,3), activation='relu'))
        # 6. Maxpooling
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        # 7. flatten output of maxPooling - Flatten()
        self.model.add(layers.Flatten())
        # 8. fully connected layer - Dense()
        # 1st hidden layer
        self.model.add(layers.Dense(units=512, activation='relu'))
        # 9. 2nd hidden layer
        self.model.add(layers.Dense(units=256, activation='relu'))
        # 10. 3rd hidden layer
        self.model.add(layers.Dense(units=128, activation='relu'))
        # 11. output layer
        self.model.add(layers.Dense(units=10, activation='sigmoid'))
        # 12. compile CNN
        # loss function(=cost function) calculates loss  -> to find best weights, have to find the lowest loss
        # optimizer -> update weights
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
        pass

    # training CNN
    def fit(self, inputs, targets, ep) :
        self.model.fit(x=inputs,y=targets, epochs=ep)
        pass

    # evaluate CNN
    def evaluate(self, inputs, targets) :
        self.model.evaluate(inputs, targets)
        pass


# data load
(inputs_training, targets_training), (inputs_test, targets_test) = tf.keras.datasets.mnist.load_data()

# data processing
inputs_training = inputs_training.reshape(inputs_training.shape[0], 28, 28, 1)
inputs_test = inputs_test.reshape(inputs_test.shape[0], 28, 28, 1)
inputs_training = inputs_training.astype('float32')
inputs_test = inputs_test.astype('float32')

# Rescaling
inputs_training /= 255
inputs_test /= 255

# building CNN
cnn = ConvolutionNetwork()

# training CNN
epochs = 5
cnn.fit(inputs_training,targets_training,epochs)

# evaluate CNN
cnn.evaluate(inputs_test, targets_test)