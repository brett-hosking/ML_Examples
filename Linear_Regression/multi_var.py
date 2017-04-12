'''
	Linear Regression with multiple variables
	Based on the MATLAB/Octave Machine Learning Coursera Assessment

	Brett Hosking
	github.com/brett-hosking/ML_Examples/Linear_Regression
'''

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def main():

	data = np.loadtxt("../data/ex1data2.txt",delimiter=',')
	x_data = data[:,[0,1]]
	y_data = data[:,2]

	# #  --- Plot test data --- #
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	ax = fig.gca(projection='3d')
	ax.scatter(x_data[:,0], x_data[:,1],y_data, cmap=cm.coolwarm)
	plt.xlabel('$x_0$'); plt.ylabel('$x_1$')
	ax.set_zlabel('y')
	plt.title('Input Data')
	plt.subplots_adjust(left=0.001,right=0.99)
	ax.view_init(35, -30)

	# ---- Python+Numpy implementation ---- #
	print "--- Linear Regression with Multiple Variables - Gradient Descent Approach ---"
	Gradient_Decent_Approach(x_data,y_data)
	plt.show()
	print "--- Linear Regression with Multiple Variables - Normal Equation Approach ---"
	Normal_Equation_Approach(x_data,y_data)
	plt.show()

def Normal_Equation_Approach(x_data,y_data):

	# size of dataset
	m = len(y_data)
	n = np.shape(x_data)[1]			# n features
	# Generate the feature matrix - add column of ones
	X = np.column_stack((np.ones((m,1)), x_data))

	# Use Normal Equation
	theta = NormEq(X,y_data)

	# Calculate the cost after fitting
	cost = Cost(X,theta,m,y_data)
	print "theta: ", theta, "\ncost: ", cost

	# plot the hypothesis with the learnt fitting values
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	ax = fig.gca(projection='3d')
	x1 = np.linspace(np.min(x_data[:,0]), np.max(x_data[:,0]), 500)
	x2 = np.linspace(np.min(x_data[:,1]), np.max(x_data[:,1]), 500)
	h = np.dot(np.column_stack((np.ones(500),np.column_stack((x1,x2)) )),theta) 
	ax.scatter(x_data[:,0], x_data[:,1],y_data, cmap=cm.coolwarm)
	ax.plot(x1,x2,h,c='r',linewidth=1.5)
	plt.xlabel('$x_0$'); plt.ylabel('$x_1$')
	ax.set_zlabel('y')
	plt.title('Linear Regression using Normal Equation')
	plt.subplots_adjust(left=0.001,right=0.99)
	ax.view_init(35, -30)


def Gradient_Decent_Approach(x_data,y_data):
	
	# size of dataset
	m = len(y_data)
	n = np.shape(x_data)[1]			# n features
	# Feature Scaling - normalise the features to ensure equal weighting
	Xnorm,mu,sigma = FeatureNorm(x_data)
	# Generate the feature matrix - add column of ones
	X = np.column_stack((np.ones((m,1)), Xnorm))
	# Initialise theta vector
	theta = np.array(np.zeros(n+1)) 
	# learning rate
	alpha = 0.01 

	# initial cost
	cost = Cost(X,theta,m,y_data)
	print "initial cost: ", cost

	# Use Gradient Descent to learn the best fitting parameters
	iterations = 400
	theta,Jhist = MultiGradientDescent(X,y_data,theta,alpha,iterations,m)
	# Calculate the cost after fitting
	cost = Cost(X,theta,m,y_data)
	print "theta: ", theta, "\ncost: ", cost

	# Plot covergence of cost
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.plot(xrange(Jhist.size), Jhist, "-b", linewidth=2 )
	plt.title("Convergence of Cost Function")
	plt.xlabel('Number of iterations')
	plt.ylabel('J($\Theta$)')


	# plot the hypothesis with the learnt fitting values
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	ax = fig.gca(projection='3d')
	x1 = np.linspace(np.min(x_data[:,0]), np.max(x_data[:,0]), 500)
	x2 = np.linspace(np.min(x_data[:,1]), np.max(x_data[:,1]), 500)
	x1norm = np.linspace(np.min(Xnorm[:,0]), np.max(Xnorm[:,0]), 500)
	x2norm = np.linspace(np.min(Xnorm[:,1]), np.max(Xnorm[:,1]), 500)
	h = np.dot(np.column_stack((np.ones(500),np.column_stack((x1norm,x2norm)) )),theta) 
	ax.scatter(x_data[:,0], x_data[:,1],y_data, cmap=cm.coolwarm)
	ax.plot(x1,x2,h,c='r',linewidth=1.5)
	plt.xlabel('$x_0$'); plt.ylabel('$x_1$')
	ax.set_zlabel('y')
	plt.title('Linear Regression using Gradient Descent')
	plt.subplots_adjust(left=0.001,right=0.99)
	ax.view_init(35, -30)

def FeatureNorm(X):

	Xnorm = np.zeros(np.shape(X))

	mu    = np.zeros((1, X.shape[1]))
	sigma = np.zeros((1, X.shape[1]))
	for i in xrange(X.shape[1]):
		mu[:,i] 	= np.mean(X[:,i])
		sigma[:,i] 	= np.std(X[:,i])
		Xnorm[:,i]	= (X[:,i] - float(mu[:,i]))/float(sigma[:,i])
	return Xnorm,mu,sigma

def NormEq(X,y):
	return np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

def MultiGradientDescent(X,y,theta,alpha,iterations,m):

	Jhist = np.zeros((iterations,1))
	xTrans = X.transpose() # use numpy version?
	for i in xrange(iterations):
		h = np.dot(X,theta)
		errors = h - np.transpose(y)  
		theta_change = (alpha/m) * np.dot(xTrans,errors)
		theta = theta - theta_change 

		Jhist[i] = Cost(X,theta,m,y)

	return theta,Jhist

def Cost(X,theta,m,y):

	h = np.dot(X,theta)
	S = np.sum((h - np.transpose(y))**2)
	J = S / (m) # or 2*m

	return J


if __name__=='__main__':
	main()