import numpy as np
import h5py
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras import backend as K

# Keras settings
K.set_image_dim_ordering('th')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
						  histogram_freq=10,
						  batch_size=10, write_graph=True,
						  write_grads=True, write_images=True,
						  embeddings_freq=0, embeddings_layer_names=None,
						  embeddings_metadata=None, embeddings_data=None)

def readData():
	'''
	This function reads in the hdf5 file - it takes
	around 3s on average to run on a
	dual processor workstation
	'''
	# read h5 format back to numpy array
	global X_train
	global Y_train
	global X_test
	h5f = h5py.File('MNISTdata.h5', 'r')
	X_train = h5f['train'][:]
	Y_train = h5f['label'][:]
	X_test = h5f['test'][:]
	h5f.close()
	print("--- Data read in successfully ---")

readData()

X_train = X_train.reshape(42000,28,28)
X_test = X_test.reshape(28000,28,28)

# Random number seed for reproducibility
np.random.seed(123)

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels
Y_train = np_utils.to_categorical(Y_train, 10)
# Y_test = np_utils.to_categorical(Y_test, 10)

# Define model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, epochs=250, verbose=1,
          callbacks=[tensorboard])

# Evaluate model on test data
# score = model.evaluate(X_test, Y_test, verbose=0)

Y_test = model.predict(X_test)

f = open("test_label.csv", 'w')
f = open("test_label.csv", 'a')
print("ImageId,Label", file=f)
for i in range(len(Y_test)):
	print("{},{}".format(i+1,np.argmax(Y_test[i])), file=f)
