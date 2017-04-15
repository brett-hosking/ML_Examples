''' 
		Single Layer Neural Network

		0-9 (Handwritten) Digit Recognition using Neural Networks
		Based on the Coursera Machine Learning course and also 
		'First Conact with TensorFlow' by Jordi Torres

		Brett Hosking
		github.com/brett-hosking/ML_Examples/Neural_Networks/
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

Display 	= True # Display a random 100 samples
Learn 		= True  # True: Learn parameters. False: Load learnt parameters

def main():

	### --- Load Data --- ###
	print "\nCollecting Data..."
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	### ---------- Display some elements from the dataset ---------- ###
	if Display:
		print "\nDisplaying selection of handwritten digits..."
		samples, _ = mnist.train.next_batch(100)	
		# Display the randomly selected digits
		displayData(samples)

	### --------------------------- NN Parameters --------------------------- ###
	Tsamples 		  = 100  # Number of training samples (out of 55000)
	# randsel			  = True # Randomly select training samples (no guarantee of an equal distribution of samples)
	# if randsel:
	# 	# Select a random set of training samples of size Tsamples
	# 	rand_indices = np.random.permutation(mnist.test.images.shape[0])
	# 	Xtrain 		 = mnist.train.images[rand_indices[0:Tsamples], :]
	# 	Ylabels		 = mnist.train.labels[rand_indices[0:Tsamples], :]
	# else:
	Xtrain, Ylabels = mnist.train.next_batch(Tsamples)	

	input_layer_size  = np.shape(Xtrain)[1]  # 28x28 Input Images of Digits
	num_labels 		  = np.shape(Ylabels)[1] # 10 labels, from 0 to 9  

	alpha			  = 0.01 # Learning Rate for gradient descent
	iterations		  = 1000 # Gradient Descent iterations
	dispsamps 		  = 5    # number of samples to display and predict

	print "\n--- NN Parameters ---\n",\
			"Trainning samples per iteration: ", np.shape(Xtrain)[0],"\n" \
			"Test samples: ", np.shape(mnist.test.images)[0], "\n" \
			"Learning rate: ", alpha, "\n" \
			"Optimise iterations: ", iterations, "\n" \
			"--------------------"

	### --------------------------- TensorFlow NN --------------------------- ###
	# -- Create TensorFlow Variables -- #
	x = tf.placeholder("float", [None, input_layer_size])			# Feature vector 	- (None indicates the dimension can be of any size)
	W = tf.Variable(tf.zeros([input_layer_size,num_labels])) 		# Weights 		    - initialised with zeros
	b = tf.Variable(tf.zeros([num_labels]))							# Bias 	  		 	- initialised with zeros

	# -- Cost Function and Optimiser -- #
	y = tf.nn.softmax(tf.matmul(x,W) + b)							# Prediction
	y_ = tf.placeholder("float", [None,num_labels])					# Placeholder for correct labels
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))					# Cost	
	train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)		

	# -- Setup TensorFlow Session -- #	
	print "Starting TensorFlow session..."		
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# -- Training -- #	
	print "Learning from training set..."
	for i in range(iterations):		
		Xtrain, Ylabels = mnist.train.next_batch(Tsamples)
		sess.run(train_step, feed_dict={x: Xtrain, y_: Ylabels})
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # compare prediction with label
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # Accuracy of prediction
	
	### --------------------------- Evaluate Model --------------------------- ###
	print "\nAccuracy on Test Set: ", sess.run(accuracy*100.0, feed_dict={x: mnist.test.images, y_: mnist.test.labels}), "%"

	# TODO: Visualise Neural Network's Hidden Layer

	# Display samples and predict
	print "\nApply model for prediction..."
	rp = np.random.permutation(mnist.test.images.shape[0])
	for i in xrange(dispsamps):
		classification  = sess.run(tf.argmax(y, 1), feed_dict={x: [mnist.test.images[rp[i], :]]})
		print('{:d}of{:d} Prediction: {:d} (actual digit {:d})'.format(i+1, dispsamps, classification[0], np.argmax(mnist.test.labels[rp[i],:]))) 
		displaydigit(mnist.test.images[rp[i],:])



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