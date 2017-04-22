''' 
		Convolutional Neural Network

		0-9 (Handwritten) Digit Recognition using 
		Convolutional Neural Networks

		Brett Hosking
		github.com/brett-hosking/ML_Examples/Deep_Learning/
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data
import tensorflow.contrib.slim as slim

Disp_set 	= True # Display a random 100 samples

Viz_example = True 	# required for the following three
# For the following it is suggested that you reduce the number of nodes in each layer that is displayed
disph1 		= True 	# Display hidden layer 1
disph2 		= True 	# Display hidden layer 2
disFC 		= True  # Display fully connected layer


def main():

	### --- Load Data --- ###
	print "\nCollecting Data..."
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	### ---------- Display some elements from the dataset ---------- ###
	if Disp_set:
		print "\nDisplaying selection of handwritten digits..."
		samples, _ = mnist.train.next_batch(100)	
		# Display the randomly selected digits
		displayData(samples)

	### --------------------------- NN Parameters --------------------------- ###
	input_layer_size  = 784  # 28x28 Input Images of Digits
	num_labels 		  = 10   # 10 labels, from 0 to 9  

	# Hyperparameters (200,1e-4,200,12,32,1024 gives ~92% Test Set accuracy)
	Tsamples 		  = 200  	# Number of training samples per iteration (out of 55000/iterations)
	alpha			  = 1e-4 	# Learning Rate for optimiser
	iterations		  = 200 	# Training iterations		
	l1_filnum		  = 12  	# layer 1 filters
	l2_filnum		  = 32 		# layer 2 filters
	FC_neurons		  = 20	# Fully Connected layer neurons


	dispsamps 		  = 5    # number of samples to display and predict

	print "\n--- CNN Parameters ---\n",\
			"Trainning samples per iteration: ", Tsamples,"\n" \
			"Learning rate: ", alpha, "\n" \
			"Optimise iterations: ", iterations, "\n" \
			"--------------------"

	### --------------------------- TensorFlow NN --------------------------- ###
	x  = tf.placeholder("float", [None, input_layer_size])		# Feature vector	- (None indicates the dimension can be of any size)
	y_ = tf.placeholder("float", [None,num_labels])				# Placeholder for correct labels
	# -- Reconstruct original image shape -- #
	x_im = tf.reshape(x,[-1,28,28,1])	# 28x28 greyscale

	# -- L1: Variables -- #
	W_conv1 = weight_variable([5,5,1,l1_filnum])				# Weights			- initialised randomly 5x5, 32 filters
	b_conv1 = bias_variable([l1_filnum])						# Bias				- 32 filters
	# -- L1: Rectified Linear Unit (ReLU) -- #
	h_conv1 = tf.nn.relu(conv2d(x_im, W_conv1) + b_conv1)
	# -- L1: Max-Pooling -- #
	h_pool1 = max_pool_2x2(h_conv1)

	# -- L2: Variables -- #
	W_conv2 = weight_variable([5, 5, l1_filnum, l2_filnum])		# Weights			- initialised randomly 5x5, 64 filters
	b_conv2 = bias_variable([l2_filnum])						# Bias				- 64 filters
	# -- L2: Rectified Linear Unit (ReLU) -- #
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	# -- L2: Max-Pooling -- #
	h_pool2 = max_pool_2x2(h_conv2)

	###############################################################
	# -- Fully-Connected Layer -- #
	# W_fc1 = weight_variable([7 * 7 * l2_filnum, FC_neurons])	# Weights 			- 7x7 as prev layer is 12x12 and the window is 5x5 with a sliding window stride of size 1
	# b_fc1 = bias_variable([FC_neurons])						# Bias 				

	# # -- Flatten for Softmax -- #
	# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*l2_filnum])
	# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# # -- Apply 'dropout' to reduce overfitting -- #
	# keep_prob = tf.placeholder("float")
	# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# -- Apply Softmas -- #
	# W_fc2 = weight_variable([FC_neurons, num_labels])
	# b_fc2 = bias_variable([num_labels])
	# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # Prediction
	##################################################################
	keep_prob = tf.placeholder("float")
	h_fc1_drop = slim.dropout(slim.conv2d(h_pool2,FC_neurons,[5,5]),keep_prob)
	y_conv = slim.fully_connected(slim.flatten(h_fc1_drop),10,activation_fn=tf.nn.softmax)

	# -- Cost Function and Optimiser -- #					
	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))		   # Cost	
	#train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)	
	train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)	

	# -- Prediction and Accuracy -- #
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  # compare prediction with label
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 	 # Accuracy of prediction

	# -- Setup TensorFlow Session -- #	
	print "Starting TensorFlow session..."		
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# -- Training -- #	
	print "Learning from training set..."
	for i in range(iterations):
		batch = mnist.train.next_batch(Tsamples)
		sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	
	### --------------------------- Evaluate Model --------------------------- ###
	#train_accuracy = sess.run( accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
	print "\nAccuracy on Training Set: %g"%sess.run( accuracy*100.0, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0}),"%"
	print "Accuracy on Test Set: %g"% sess.run(accuracy*100.0, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}),"%"

	# TODO: Visualise Neural Network's Hidden Layer
	if Viz_example:
		print "Visualisation of hidden layers..."
		digit_example = mnist.test.images[int(np.random.uniform(low=0.0, high=len(mnist.test.images), size=1))]
		displaydigit(digit_example)
		if disph1: dispActivations(h_conv1,digit_example,sess,x,keep_prob)
		if disph2: dispActivations(h_conv2,digit_example,sess,x,keep_prob)
		if disFC:  dispActivations(h_fc1_drop,digit_example,sess,x,keep_prob)

	# Display samples and predict
	# print "\nApply model for prediction..."
	# rp = np.random.permutation(mnist.test.images.shape[0])
	# for i in xrange(dispsamps):
	# 	classification  = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [mnist.test.images[rp[i], :]]})
	# 	print('{:d}of{:d} Prediction: {:d} (actual digit {:d})'.format(i+1, dispsamps, classification[0], np.argmax(mnist.test.labels[rp[i],:]))) 
	# 	displaydigit(mnist.test.images[rp[i],:])

def dispActivations(layer,stimuli,sess,x,keep_prob):
	units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
	plotNNFilter(units)

def plotNNFilter(units):
	filters = units.shape[3]
	plt.figure(1, figsize=(20,20))
	n_columns = 6
	n_rows = np.ceil(filters / n_columns) + 1
	for i in range(filters):
		plt.subplot(n_rows, n_columns, i+1)
		#plt.title('Filter ' + str(i))
		plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
	plt.show()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def displaydigit(pixels,dim=[28,28]):

	digit = np.reshape(pixels, (dim[0],dim[1]))
	plt.imshow(digit,cmap='gray')
	plt.show()


def displayData(X):

	# Create a greyscale image of the digits (m^(1/2))x(m^(1/2))
	example_width = int(round(np.sqrt(np.size(X, 1))))

	# samples, features
	[m,n] = np.shape(X)

	example_height = (n/example_width)

	# display grid dim
	disp_rows = np.floor(np.sqrt(m))
	disp_cols = np.ceil(m / disp_rows)

	# padding between images
	pad = 1

	# Array to display
	display_area = np.array(np.ones((int(pad + disp_rows * (example_height + pad)),
									int(pad + disp_cols * (example_width + pad)	 ))) )

	

	# wrap each example into a patch
	cur_ex = 0 
	for j in xrange(int(disp_rows)):
		for i in xrange(int(disp_cols)):

			# patch data
			patch = np.reshape(X[cur_ex,:],(example_height,example_width))

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