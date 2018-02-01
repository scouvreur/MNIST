import numpy as np
import h5py
from matplotlib import pyplot as plt

def loadData():
	'''
    This function reads in all the CSV data and saves it to an
    hdf5 file - it takes around 1min on average to run on a
    dual processor workstation. Each image is a 1D array with length
    784, the pixel intensity values are integers from 0 to 255. We can
    therefore use unsigned 8-bit integers.
	'''
	# train data format
	# label,pixel0,pixel1,pixel2 [...] ,pixel783
	# 1,0,0,0, [...] ,255,0

	# test data format
	# pixel0,pixel1,pixel2 [...] ,pixel783
	# 0,0,0, [...] ,255,0
	global train
	global label
	global test
	train = np.loadtxt("train.csv", delimiter=',', skiprows=1, usecols=range(1,785), dtype='uint8')
	label = np.loadtxt("train.csv", delimiter=',', skiprows=1, usecols=(0), dtype='uint8')
	test = np.loadtxt("test.csv", delimiter=',', skiprows=1, usecols=range(0,784), dtype='uint8')
	print("---  Data loaded successfully ---")

def saveData():
	'''
	This function writes all the data to an hdf5 file
	'''
	# write numpy arrary tensor into h5 format
	h5f = h5py.File('MNISTdata.h5', 'w')
	h5f.create_dataset('train', data=train)
	h5f.create_dataset('label', data=label)
	h5f.create_dataset('test', data=test)
	h5f.close()

def readData():
	'''
	This function reads in the hdf5 file - it takes
	around 3s on average to run on a
	dual processor workstation
	'''
	# read h5 format back to numpy array
	global train
	global label
	global test
	h5f = h5py.File('MNISTdata.h5', 'r')
	train = h5f['train'][:]
	label = h5f['label'][:]
	test = h5f['test'][:]
	h5f.close()
	print("--- Data read in successfully ---")

def showImageWhite(i):
	'''
	This function takes in an integer index of the image in the 42000
	MNIST training dataset and plots its white on black image in matplotlib.
	'''
	i = int(i)
	plt.title('Label is {}'.format(label[i]))
	plt.axis('off')
	plt.imshow(train[i].reshape(28,28), interpolation='nearest', cmap='gray')
	plt.show()

def showImageBlack(i):
	'''
	This function takes in an integer index of the image in the 42000
	MNIST training dataset and plots its black on white image in matplotlib.
	'''
	i = int(i)
	plt.title('Label is {}'.format(label[i]))
	plt.axis('off')
	plt.imshow(255 - train[i].reshape(28,28), interpolation='nearest', cmap='gray')
	plt.show()
