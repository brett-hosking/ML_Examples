'''
	Linear Regression with one variable using Gradient Descent 

		Steps:
				1/ Calculate the Hypothesis: h = X * theta
				2/ Calculate the error: err = h - y
				3/ Calculate the gradient: (X' * err)/ m 
				4/ Update the fitting parameters: theta = theta - alpha * gradient

			where: 
			h is the hypothesis, 
			m is length of the dataset, 
			X is a matrix of (n x m) where n is the number of features (theta0*1 + theta1*x)
			theta is a vector of length n which contains the fitting parameters,
			alpha is the learning rate


'''

'''
TODO: Add scikit-learn and TensorFlow examples
#import tensorflow as tf
#import sklearn
'''
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def main():

	# --- Generate Data --- #
	num_points = 1000
	x_data,y_data = [],[]

	# Define initial theta parameters - Gradient Descent should produce similar values
	t0,t1 = 0.4,0.2

	for i in xrange(num_points):
		x = np.random.normal(0.0,0.55)
		y = t0 + (x * t1) + np.random.normal(0.0,0.03)
		x_data.append(x)
		y_data.append(y)

	# #  --- Plot test data --- #
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.plot(x_data,y_data, 'ro')
	plt.title('Test Data')
	plt.xlabel('x')
	plt.ylabel('y')

	# ---- Python+Numpy implementation ---- #
	print "--- Linear Regression using Python + Numpy ---"
	# size of dataset
	m = len(y_data)
	# Generate the feature matrix
	X = np.transpose(np.array([np.ones(m),x_data])) 
	# Initialise theta vector
	theta = np.array(np.zeros(2)) 
	# learning rate
	alpha = 0.01 

	# initial cost
	cost = Cost(X,theta,m,y_data)
	print "initial cost: ", cost

	# Use Gradient Descent to learn the best fitting parameters
	iterations = 1500
	theta = GradientDescent(X,y_data,theta,alpha,iterations,m)
	# Calculate the cost after fitting
	cost = Cost(X,theta,m,y_data)
	print "theta: ", theta, " cost: ", cost

	# plot the hypothesis with the learnt fitting values
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	h = np.dot(X,theta) 
	plt.plot(x_data,y_data, 'ro')
	plt.plot(x_data,h, label="linear regression")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Linear Regression using Gradient Descent')
	plt.legend()


	# Generate cost surface 
	# generate theta values and produce a 2D meshgrid
	theta0_vals = np.linspace(t0-5,t0+5,200)
	theta1_vals = np.linspace(t1-2,t1+2,200)
	T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

	# initialize J_vals to a matrix of 0's
	J_vals = np.array(np.zeros((len(theta1_vals),len(theta0_vals))))

	# Calculate the cost for each theta0 theta1 combination
	for j in xrange(len(theta1_vals)):
		for i in xrange(len(theta0_vals)):
			t = np.array([theta0_vals[i],theta1_vals[j]])   
			J_vals[j,i] = Cost(X,t,m,y_data)


	# Plot Surface
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(T0, T1, J_vals, cmap=cm.coolwarm)
	plt.xlabel('$\Theta_0$'); plt.ylabel('$\Theta_1$')
	ax.set_zlabel('J($\Theta$)')
	plt.title('Cost Surface')
	plt.subplots_adjust(left=0.001,right=0.99)

	# Plot cost contours
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	CS = plt.contour(theta0_vals, theta1_vals, J_vals,np.logspace(-1, 1, 20))
	plt.scatter(theta[0],theta[1],marker='x',color='r',s=50)
	plt.clabel(CS, inline=1, fontsize=10)	
	plt.xlim(t0-5,t0+5)
	plt.ylim(t1-2,t1+2)
	plt.xlabel('$\Theta_0$'); plt.ylabel('$\Theta_1$')
	plt.title('Cost Surface J($\Theta$)')

	plt.show()

def GradientDescent(X,y,theta,alpha,iterations,m):
	xTrans = X.transpose() # use numpy version?
	for i in xrange(iterations):

		h = np.dot(X,theta)
		errors = h - y 
		theta_change = (alpha/m) * np.dot(xTrans,errors)
		theta = theta - theta_change 

	return theta

def Cost(X,theta,m,y):

	h = np.dot(X,theta)
	S = np.sum((h - y)**2)
	J = S / (2*m)

	return J


if __name__=='__main__':
	main()