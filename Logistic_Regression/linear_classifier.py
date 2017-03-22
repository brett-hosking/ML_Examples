'''
		Logistic Regression using a linear classifier 
		Based on the MATLAB/Octave Machine Learning Coursera Assessment

		Brett Hosking
		github.com/brett-hosking/ML_Examples/Logistic_Regression
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def main():

	# ---- Import Data ---- #
	data = np.loadtxt("ex2data1.txt",delimiter=',')
	x_data = data[:,[0,1]]
	y_data = data[:,2]

	# -- Compute cost and gradient -- #
	m = float(np.shape(x_data)[0])	# m training samples
	n = np.shape(x_data)[1]			# n features
	
	# add column of 1s
	X = np.column_stack((np.ones(int(m)),x_data))

	# Initial theata
	theta = np.zeros(n+1)

	# Compute and display initial cost and gradient
	cost = costFunc(theta,X,y_data,m)
	grad = gradFunc(theta,X,y_data,m)

	print '\nCost at initial theta (zeros): ', cost
	print 'Gradient at initial theta (zeros): ', grad, '\n'

	# --- Optimise using (fminunc alternative) --- #
	theta_opt = opt.fmin_bfgs(costFunc, theta, fprime=gradFunc, args=(X, y_data,m),disp=False)
	cost = costFunc(theta_opt,X,y_data,m) 

	print '\nCost at theta at optimum: ', cost
	print 'theta: ', theta_opt, '\n'

	# Calculate the desicision boundary
	xpoints = np.linspace(min(x_data[:,0])-2,max(x_data[:,0]+2),num=m+4)
	boundary = decision_boundary(xpoints,theta_opt)

	# --- Plot Decision Boundary --- #
	pos,neg = np.where(y_data==1),np.where(y_data==0)
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.scatter(x_data[pos,0],x_data[pos,1], marker='o',c='r')
	plt.scatter(x_data[neg,0],x_data[neg,1], marker='x',c='b')
	plt.plot(xpoints,boundary,linewidth=1.5,c='k')
	plt.xlim([min(x_data[:,0]-2),max(x_data[:,0])+2])
	plt.ylim([min(x_data[:,1]-2),max(x_data[:,1])+2])
	plt.xlabel("$x_1$")
	plt.ylabel("$x_2$")
	plt.title("Logisitic Regression")
	plt.show()

def decision_boundary(xpoints,theta):
	# plot the decision boundary between the two classes
	# the line is of the form y=mx+c where y = x2
	# theta0 + theta1.x + theta2.x2 >=0
	# rearange to make x2 the subject:
	return -np.divide(theta[0],theta[2]) - np.multiply(np.divide(theta[1],theta[2]),xpoints)

def costFunc(theta,X,y,m):

	H = sigmoid(np.dot(X,theta))
	return (1 / m) * np.sum( -np.transpose(y)*np.log(H) - np.transpose(1-y)*np.log(1 - H) )


def gradFunc(theta,X,y,m):

	H = sigmoid(np.dot(X,theta))
	return (1/m) * np.dot(np.transpose(X),(H - y)) 
	

def sigmoid(z):

	return np.divide(1.0, (np.add(1.0,np.exp(-z)) ) )


if __name__=='__main__':
	main()