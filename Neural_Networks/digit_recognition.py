''' 
		0-9 (Handwritten) Digit Recognition using Neural Networks
		Partly based on the MATLAB code produced for the Coursera Machine Learning course

		This example uses parameters that are already trained to demonstrate the feedforward
		algorithm.

		Brett Hosking
		github.com/brett-hosking/ML_Examples/Neural_Networks/
'''

import numpy as np
import scipy.optimize as opt
import scipy.io as sio

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Display = True # Display a random 100 samples

def main():

	### --- Generate Data --- ###
	# Load MAT file
	mat = sio.loadmat("../data/digit_data")
	'''	
		5000 Trainning examples
		20x20 pixels in each (400)
		Headers: 'X'(5000,400), 'y'(5000)
	'''
	X = mat['X']
	y = mat['y'].flatten()

	# Size of dataset
	m = np.shape(X)[0] # samples
	n = np.shape(X)[1] # features 

	### ---------- Display some elements from the dataset ---------- ###
	if Display:
		# Randomly select 100 data points to display
		rand_indices = np.random.permutation(m);
		sel = X[rand_indices[0:100], :]

		# Display the randomly selected digits
		displayData(sel)
	### ------------------------------------------------------------ ###

	### --------------------------- Neural Network --------------------------- ###
	#% Setup the parameters you will use for this exercise
	input_layer_size  = 400  # 20x20 Input Images of Digits
	hidden_layer_size = 25   # 25 hidden units
	num_labels 		  = 10   # 10 labels, from 1 to 10   
	                         # (note that "0" is mapped to label 10 -> use 10%)
	
	# Load weights using calculated using the backpropagation algorithm
	weights = sio.loadmat('../data/NN_digit_recog_weights')
	Theta1,Theta2 = weights['Theta1'],weights['Theta2']
	'''	
		2 Theta matricies, one for the hidden layer and one for the output layer

		Theta1: (25,401) - 25 units in the hidden layer. 400 features (pixels) plus the bias unit.
		Theta2: (10,26)  - 10 units (classes) in the output layer. 25 + bias from previous layer
	'''

	# Feedforward and Prediction
	pred = predict(Theta1,Theta2,X)
	print "Accuracy: ", np.mean(pred+1 == y)*100.0, "%" 

	# Display samples and predict
	rp = np.random.permutation(X.shape[0])
	dispsamps = 10 # number of samples to display
	for i in xrange(dispsamps):
		pred = predict(Theta1, Theta2, X[rp[i], :])
		print('Neural Network Prediction: {:d} (actual digit {:d})'.format((pred[0]+1)%10, y[rp[i]]%10)) 
		displaydigit(rp[i], X)
	### ---------------------------------------------------------------------- ###

def predict(Theta1,Theta2,X):
	if X.ndim == 1:
		X = np.reshape(X, (-1,X.shape[0]))
	a1 = np.column_stack((np.ones(np.shape(X)[0]),X))
	a2 = np.column_stack((np.ones(np.shape(X)[0]),sigmoid(a1.dot(Theta1.T))))
	a3 = sigmoid(a2.dot(Theta2.T))
	return np.argmax(a3, axis=1) 

def sigmoid(z):

	return np.divide(1.0, (np.add(1.0,np.exp(-z)) ) )

def displaydigit(idx,X):

	pixels = X[idx,:]
	digit = np.reshape(pixels, (20,20),order='F')
	plt.imshow(digit,cmap='gray')
	plt.show()


def displayData(X):

	# Create a greyscale image of the digits (m^(1/2))x(m^(1/2))
	example_width = round(np.sqrt(np.size(X, 1)))

	# samples, features
	[m,n] = np.shape(X)

	example_height = (n/example_width)

	# display grid dim
	disp_rows = np.floor(np.sqrt(m))
	disp_cols = np.ceil(m / disp_rows)

	# padding between images
	pad = 1

	# Array to display
	display_area = np.array(np.ones((pad + disp_rows * (example_height + pad),
									pad + disp_cols * (example_width + pad)	 )) )

	# wrap each example into a patch
	cur_ex = 0 
	for j in xrange(int(disp_rows)):
		for i in xrange(int(disp_cols)):

			# patch data
			patch = np.reshape(X[cur_ex,:],(example_height,example_width),order="F")

			# load into array
			yidx = ((j+1)*pad) + (example_height*j) 
			xidx = ((i+1)*pad) + (example_width*i)
			display_area[yidx:yidx+example_height,xidx:xidx+example_width] = patch

			cur_ex+=1
			if cur_ex > m:
				break

	# Display as a Matplotlib image
	plt.imshow(display_area,cmap='gray')
	plt.show()

if __name__=='__main__':
	main()