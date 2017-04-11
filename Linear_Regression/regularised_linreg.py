'''
	Regularised Linear Regression
	Based on the MATLAB/Octave Machine Learning Coursera Assessment

	Brett Hosking
	github.com/brett-hosking/ML_Examples/Linear_Regression
'''

import numpy as np 
import matplotlib.pyplot as plt

import scipy.io as sio
import scipy.optimize as opt


def main():

	# Load MAT file
	mat = sio.loadmat("ex5data1")

	# Training Data
	'''	
		12 Trainning examples
		1 feature
		X: (12,1), y: (12,1)
	'''
	Xdata = mat['X']
	y = mat['y'].flatten()

	# Validation and Test Data
	# (21,1)
	Xval  = mat['Xval']
	yval  = mat['yval'].flatten()
	Xtest = mat['Xtest']
	ytest = mat['ytest'].flatten()


	# Size of dataset
	m     = np.shape(Xdata)[0] # samples
	mtest = np.shape(Xtest)[0] # samples in Xval and Xtest sets
	n     = np.shape(Xdata)[1] # features 

	# Feature Matrix
	X     = np.column_stack((np.ones(m),Xdata))
	Xval  = np.column_stack((np.ones(mtest),Xval))
	Xtest = np.column_stack((np.ones(mtest),Xtest))

	#  --- Plot test data --- #
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.plot(Xdata,y, 'ro')
	plt.title('Test Data')
	plt.xlabel('Change in water level (x)') 
	plt.ylabel('Water flowing out of the dam (y)')
	# ----------------------- #

	# --- Test Cost Function --- #
	theta = [1 , 1]
	J,grad = linearRegCostFunction(theta,X, y, 1,m)
	print 'Cost at theta = [1 ; 1]: (this value should be about 303.993192)\n', J
	print 'Gradient at theta = [1 ; 1]: (this value should be about [-15.303016; 598.250744])\n', grad
	# -------------------------- #

	#  -- Train on data and plot result with Lambda=0 -- #
	theta = train(X,y,0,m)
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	h = np.dot(X,theta) 
	plt.plot(Xdata,y, 'ro')
	plt.plot(Xdata,h,'b--')
	plt.xlabel('Change in water level (x)') 
	plt.ylabel('Water flowing out of the dam (y)')
	plt.title('Regularised Linear Regression, $\lambda=0$')
	# -- --------------------------------------------- -- #

	# -- Calculate and show learning curve -- #
	Lambda = 0
	error_train, error_val = learning_curve(X, y, Xval, yval, Lambda,m)
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	sample_range = np.arange(1,m+1)
	plt.plot(sample_range,error_train,linewidth=1.5,label='Train')
	plt.plot(sample_range,error_val,linewidth=1.5,label='Cross Validation')
	plt.xlabel('Number of Training Samples')
	plt.ylabel('Error')
	plt.title('Learning Curve for Linear Regression')
	plt.legend(loc="best", prop={'size':9})
	# -- --------------------------------- -- #


	# -- Feature Mapping for Polynomial Regression -- #
	p = 8

	# Map X onto Polynomial Features and Normalize
	X_poly = PolyFeatures(X, p)
	X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize

	# # Map X_poly_test and normalize (using mu and sigma)
	X_poly_test = PolyFeatures(Xtest, p)
	X_poly_test[:,1:] = X_poly_test[:,1:] - mu[1:]
	X_poly_test[:,1:] = X_poly_test[:,1:] / sigma[1:]

	# # Map X_poly_val and normalize (using mu and sigma)
	X_poly_val = PolyFeatures(Xval, p)
	X_poly_val[:,1:] = X_poly_val[:,1:] - mu[1:]
	X_poly_val[:,1:] = X_poly_val[:,1:] / sigma[1:]

	print('Normalized Training Example 1:')
	print('  {:s}  '.format(X_poly[0, :]))
	# -- ----------------------------------------- -- #

	# -- Learning Curve for Polynomial Regression -- #
	lambda_val = 0.5 # change this value to trade-off under/over -fitting
	theta = train(X_poly, y, lambda_val,m)

	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	plt.plot(Xdata,y, 'ro')
	# Polynomial Plot
	n_points = 50
	xvals = np.linspace(np.min(X) - 10,np.max(X) + 10,n_points)
	xmat = np.ones((n_points,1))

	xmat = np.insert(xmat,xmat.shape[1],xvals.T,axis=1)
	xmat = PolyFeatures(xmat,len(theta)-2)
	#reversing normalisation of features
	xmat[:,1:] = xmat[:,1:] - mu[1:]
	xmat[:,1:] = xmat[:,1:] / sigma[1:]

	plt.plot(xvals,np.dot(xmat,theta) ,'b--')
	plt.xlabel('Change in water level (x)') 
	plt.ylabel('Water flowing out of the dam (y)')
	plt.title ('Polynomial Regression Fit ($\lambda$ = {:.2f})'.format(lambda_val))

	# Learning Curve
	fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
	error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, lambda_val,m)
	p1, p2 = plt.plot(np.arange(1,m+1), error_train, np.arange(1,m+1), error_val)

	plt.title('Polynomial Regression Learning Curve (lambda = {:.2f})'.format(lambda_val))
	plt.xlabel('Number of training examples')
	plt.ylabel('Error')
	plt.axis([0, 13, 0, 100])
	plt.legend((p1, p2), ('Train', 'Cross Validation'))

	print('Polynomial Regression (lambda = {:.2f})\n\n'.format(lambda_val))
	print('# Training Examples\tTrain Error\tCross Validation Error\n')
	for i in xrange(m):
	    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))
	# -- ---------------------------------------- -- #


	# -- Validation for Selecting Lambda -- #

	# -- ------------------------------- -- #

	plt.show()

def PolyFeatures(Xin,p):
    Xpoly = Xin.copy()
    for i in xrange(p):
        Xpoly = np.insert(Xpoly,Xpoly.shape[1],np.power(Xpoly[:,1],i+2),axis=1)
    return Xpoly

def featureNormalize(Xin):   
    Xnorm = Xin.copy()
    means = np.mean(Xnorm,axis=0) 
    Xnorm[:,1:] = Xnorm[:,1:] - means[1:]
    sigmas = np.std(Xnorm,axis=0,ddof=1)
    Xnorm[:,1:] = Xnorm[:,1:] / sigmas[1:]
    return Xnorm, means, sigmas

def learning_curve(X, y, Xval, yval, Lambda,m):
	# Error/Cost vs number of training samples
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))

    for i in xrange(1,m+1):

        X_train = X[:i]
        y_train = y[:i]

        theta = train(X_train, y_train, Lambda,m)

        error_train[i-1],_ = linearRegCostFunction(theta, X_train, y_train,   0, np.shape(X_train)[0])
        error_val[i-1],_   = linearRegCostFunction(theta, Xval, 	 yval,    0, np.shape(Xval)[0])
                
    return error_train, error_val

def train(X,y,Lambda,m):
	init_theta = np.zeros((X.shape[1], 1))
	args=(X,y,Lambda,m)
	return opt.minimize(linearRegCostFunction, x0=init_theta, args=args, options={'disp': False, 'maxiter':200}, method="L-BFGS-B", jac=True)["x"]

def linearRegCostFunction(theta,X, y, Lambda,m):

	# Regularised Linear Regression Cost Function
	H = np.dot(X,theta) 
	#J = ( 1./(2*m)) * np.power( (H - y) , 2).sum() + ( float(Lambda) / (2*m)) * np.power(theta[1:],2).sum()
	J = (1./(2*m))*np.sum(np.square(H-y)) + (float(Lambda)/(2*m))*np.sum(np.square(theta[1:]))

	# Regularised Linear Regression Gradient
	theta_reg = np.array([np.insert(theta[1:len(theta)],0,0)])
	grad = (1./m) * np.dot( X.T, H - y) + ( float(Lambda) / m )*theta_reg

	return J,grad[0]

if __name__=='__main__':
	main()