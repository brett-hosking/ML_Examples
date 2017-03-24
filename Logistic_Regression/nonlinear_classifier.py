'''
		Logistic Regression using a non-linear classifier 
		Based on the MATLAB/Octave Machine Learning Coursera Assessment

		Brett Hosking
		github.com/brett-hosking/ML_Examples/Logistic_Regression
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def main():

	# ---- Import Data ---- #
	data = np.loadtxt("ex2data2.txt",delimiter=',')
	x_data = data[:,[0,1]]
	y_data = data[:,2]

	# Feature Mapping - Add polynomial features
	X = featuremapping(x_data[:,0],x_data[:,1],degree=6)

	# -- Training set and Feauture set size-- #
	m = float(np.shape(X)[0])	# m training samples
	n = np.shape(X)[1] -1		# n features

	# Initial theta
	theta = np.zeros(n+1)

	# Set regularization parameter lambda to 1
	Lambda = 1

	cost = cost_reg(theta,X,y_data,Lambda)

	print 'Cost at initial theta (zeros): \n', cost

	# --- Optimise using (fminunc alternative) --- #
	theta_opt = opt.fmin_bfgs(cost_reg, theta, args=(X, y_data,Lambda))
	cost = cost_reg(theta_opt,X,y_data,Lambda) 

	print '\nCost at theta at optimum: ', cost

	# Calculate the desicision boundary
	grid = [np.linspace(-1, 1.5, 50),np.linspace(-1, 1.5, 50)]
	z = nonlin_decision_boundary(theta_opt,grid)

	# Point for classification
	point = [0,0.25]
	# Probability of point being part of class 1
	prob = sigmoid(np.dot(featuremapping(point[0], point[1])[0], theta_opt))
	print "Probability of new data point being part of class 1: ", prob

	# --- Plot Decision Boundary --- #
	pos,neg = np.where(y_data==1),np.where(y_data==0)
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.scatter(x_data[pos,0],x_data[pos,1], marker='o',c='r',label='class 1')
	plt.scatter(x_data[neg,0],x_data[neg,1], marker='x',c='b', label='class 2')
	plt.contour(grid[0], grid[1], z, levels=[0], linewidth=2,label="Decision Boundary").collections[0]
	plt.xlim([min(x_data[:,0]-0.25),max(x_data[:,0])+0.25])
	plt.ylim([min(x_data[:,1]-0.25),max(x_data[:,1])+0.25])
	plt.scatter(point[0],point[1],marker='*',c='green',s=50,label='new point')
	plt.xlabel("$x_1$")
	plt.ylabel("$x_2$")
	plt.title("Logisitic Regression")
	plt.legend(loc='best', prop={'size':9},scatterpoints=1)
	plt.show()

	# Predict Accuracy on Training Set
	print "Accuracy on Training Set: ", accuracy(theta_opt,X,y_data), "%"

def featuremapping(X1,X2,degree=6):
	#   Feature mapping function to polynomial features
	try:
		out = np.ones(( X1.shape[0], sum(range(degree + 2)) )) 
	except:
		out = np.ones(( 1, sum(range(degree + 2)) )) 

	idx = 1
	for i in xrange(1, degree + 1):
		for j in xrange(i+1):
			out[:,idx] = np.power(X1,i-j) * np.power(X2,j)
			idx += 1

	return out


def cost_reg(theta,X,y,Lambda):

		m = float(len(y))
		H = sigmoid(np.dot(X,theta))

		T = np.multiply(y,np.log(H)) + np.multiply(np.subtract(1,y),(np.log(1-H)))
		J = -np.dot((1.0/m), np.sum(T)) + np.dot(Lambda/(2*m), np.sum(np.square(theta[1:len(theta)])))
		return J

def grad_reg(theta,X,y,Lambda):

	m = float(len(y))
	H = sigmoid(np.dot(X,theta))

	ta = np.transpose(np.array([np.insert(theta[1:len(theta)],0,0)]))
	grad = np.add(np.dot(np.transpose(X),(H - y))/m, np.multiply((Lambda/m),ta)) 

	return grad

def nonlin_decision_boundary(theta,grid):

	z = np.zeros(( len(grid[0]), len(grid[1]) ))
	# Evaluate z = theta*x over the grid
	for i in xrange(len(grid[0])):
	    for j in xrange(len(grid[1])):
	        z[i,j] = np.dot(featuremapping(np.array([grid[0][i]]), np.array([grid[1][j]])),theta)
	z = np.transpose(z) 
	
	return z

def accuracy(theta,X,labels):
	p = sigmoid(np.dot(X, theta))>=0.5 
	return np.mean(p == labels) * 100.0
	

def sigmoid(z):

	return np.divide(1.0, (np.add(1.0,np.exp(-z)) ) )


if __name__=='__main__':
	main()