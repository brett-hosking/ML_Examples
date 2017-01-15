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

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import tensorflow as tf
from sklearn.linear_model import SGDRegressor 

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
	Python_Example(x_data,y_data)

	# --- TensorFlow Implementation --- #
	print "--- Linear Regression using TensorFlow ---"
	TensorFlow_Example(x_data,y_data)

	print "--- Linear Regression using scikit-learn ---"
	Scikit_Learn_Example(x_data,y_data)

	plt.show()

def Scikit_Learn_Example(x_data,y_data):

	# size of dataset
	m = len(y_data)
	# Generate the feature matrix
	X = np.transpose(np.array([np.ones(m),x_data])) 

	# Initialise theta vector
	theta = np.array(np.zeros(2)) 

	# learning rate
	alpha = 0.01
	# No. iterations
	iterations = 1500

	# Create and fit the model
	model = SGDRegressor(loss='squared_loss',n_iter=iterations,
						learning_rate='constant',eta0=alpha )
	model.fit(X, y_data, coef_init=theta)

	#plot the linear regression line with the data
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.plot(x_data, y_data,'ro')
	plt.plot(x_data,model.predict(X),label="linear regression")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Linear Regression using scikit-learn\'s Gradient Descent')
	plt.legend()
	# plt.show()

def TensorFlow_Example(x_data,y_data):

	# initialise tensorflow variable classes
	# Setup variables
	t0 = tf.Variable(tf.zeros([1]))
	t1 = tf.Variable(tf.zeros([1])) 
	h = t1 * x_data + t0

	# Cost function is the MSE
	cost = tf.reduce_mean(tf.square(h-y_data))
	# learning rate
	alpha = 0.01
	# No. iterations
	iterations = 1500

	optimiser = tf.train.GradientDescentOptimizer(alpha)
	train = optimiser.minimize(cost)

	# Initialise tf variables and create session
	init = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)

	# Run Gradient Descent
	for step in xrange(iterations):
		sess.run(train)

	# print "theta: ", (step,sess.run(t0),sess.run(t1))
	# print "Cost:", (step,sess.run(cost))

	#plot the linear regression line with the data
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.plot(x_data, y_data,'ro')
	plt.plot(x_data,sess.run(t1)* x_data + sess.run(t0),label="linear regression")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Linear Regression using TensorFlow\'s Gradient Descent')
	plt.legend()
	#plt.show()

def Python_Example(x_data,y_data):
	
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
	# cost = Cost(X,theta,m,y_data)
	# print "theta: ", theta, " cost: ", cost

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
	theta0_vals = np.linspace(theta[0]-5,theta[0]+5,200)
	theta1_vals = np.linspace(theta[1]-2,theta[1]+2,200)
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
	plt.xlim(theta[0]-4,theta[0]+4)
	plt.ylim(theta[1]-2,theta[1]+2)
	plt.xlabel('$\Theta_0$'); plt.ylabel('$\Theta_1$')
	plt.title('Cost Surface J($\Theta$)')

	#plt.show()

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
	J = S / (m) # or 2*m

	return J


if __name__=='__main__':
	main()