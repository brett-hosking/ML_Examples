''' 
		0-9 (Handwritten) Digit Recognition using Logistic Regression
		Partly based on the MATLAB code produced for the Coursera Machine Learning course

		This example uses the 'One Vs All' method to create 10 Logisitic Regression classifiers, one 
		for each digit 0-9. The accuracy is then calculated on the training set. 
		Note that practically you would test the performance of your classifiers on a new set of data,
		not your training data.

		Brett Hosking
		github.com/brett-hosking/ML_Examples/Logistic_Regression
'''

import numpy as np
import scipy.optimize as opt
import scipy.io as sio

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
	y = mat['y'].flatten() # y should of of shape (m,) NOT (m,1) - the scipy optimiser will not work otherwise

	# Size of dataset
	m = np.shape(X)[0] # samples
	n = np.shape(X)[1] # features 

	### ---------- Display some elements from the dataset ---------- ###
	# Randomly select 100 data points to display
	rand_indices = np.random.permutation(m);
	sel = X[rand_indices[0:100], :]

	# Display the randomly selected digits
	displayData(sel)
	### ------------------------------------------------------------ ###


	### ----------------------- Scikit Learn ----------------------- ###
	print "\nScikit Learn Example"
	from sklearn.linear_model import LogisticRegression
	logreg = LogisticRegression(C=10, penalty='l2', solver='liblinear',intercept_scaling=0.1)
	logreg.fit(X,y.ravel())
	pred2 = logreg.predict(X)
	print('Accuracy on Training Set: {} %'.format(np.mean(pred2 == y.ravel())*100))
	### ------------------------------------------------------------ ###


	### ------------- Vectorise Logistic Regression ------------- ###
	print "\nPython + Numpy + Scipy Example"
	# Add ones to X matrix
	X = np.column_stack((np.ones(m),X))
	num_labels = 10
	Lambda = 0.1
	all_theta = oneVall(X,y,num_labels,Lambda,m,n)
	print "Accuracy on Training Set: ", np.mean(pred_oneVall(all_theta,X,y) == y%10)*100.0, "%" 
	### --------------------------------------------------------- ###


def pred_oneVall(all_theta,X,labels):
	return np.argmax(sigmoid( np.dot(X,all_theta.T) ), axis=1)


def costFunc(theta,X,y,m,Lambda):
	H = sigmoid(np.dot(X,theta))
	T = y * np.transpose(np.log( H )) + (1-y) * np.transpose(np.log( 1 - H ))
	reg = ( float(Lambda) / (2*m)) * np.power(theta[1:theta.shape[0]],2).sum()
	return -(1./m)*T.sum() + reg


def gradFunc(theta,X,y,m,Lambda):
	theta_reg = np.array([np.insert(theta[1:len(theta)],0,0)])
	H = sigmoid(np.dot(X,theta))
	grad = (1./m) * np.dot(H.T - y, X).T + ( float(Lambda) / m )*theta_reg
	return grad.flatten()


def oneVall(X,y,num_labels,Lambda,m,n):
	all_theta = np.array(np.zeros((num_labels,n+1)))

	for c in xrange(num_labels):
		initial_theta = np.zeros((n+1,1))
		args = (X,(y%10==c).astype(int),float(m),Lambda)
		all_theta[c,:] = opt.minimize(costFunc, x0=initial_theta, args=args, options={'disp': False, 'maxiter':13}, method="Newton-CG", jac=gradFunc)["x"]

	return all_theta

def sigmoid(z):

	return np.divide(1.0, (np.add(1.0,np.exp(-z)) ) )

def displayData(X):
	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg

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