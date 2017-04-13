'''
	Support Vector Machines (SVM)
	Based on the MATLAB/Octave Machine Learning Coursera Assessment

	This example uses scikit-learn to learn the SVM hypothesis 

	Brett Hosking
	github.com/brett-hosking/ML_Examples/Support_Vector_Machines
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
from sklearn import svm 

def main():

	Data1 = True
	Data2 = True

	if Data1:
		### --------- Data Set 1 --------- ###
		# Load MAT file
		mat = sio.loadmat("../data/ex6data1")
		X, y = mat['X'], mat['y'] 
		'''
			X: (51,2)
			y: (51,1)
		'''
		# Organise data into classes
		#c1,c2 = np.where(y==1),np.where(y==0)
		c1,c2 = (y == 1).ravel(),(y == 0).ravel()

		# Train the SVM using the C-Support Vector Classification
		'''
			The C parameter is a positive value that controls the penalty for the misclassified
			trainng examples. A large value intructs the SVM to classify all data points correctly.
		'''
		Csup = 1.0 # ~30 will produce a boundary with all points classified correctly
		clf = svm.SVC(C=Csup, kernel='linear')
		clf.fit( X, y.flatten() ) 

		# Generate arrays to plot the decision boundary
		x1points,x2points,z = decision_boundary(clf, X)

		# Plot data with decision boundary
		fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
		plt.scatter(X[c1,0], X[c1,1], marker='o',c='r',label='class 1')
		plt.scatter(X[c2,0], X[c2,1], marker='x',c='b',label='class 2')
		plt.contour(x1points, x2points, z, levels=[0], linewidth=2)
		plt.xlim([min(X[:,0]-0.25),max(X[:,0])+0.25])
		plt.ylim([min(X[:,1]-0.25),max(X[:,1])+0.25])
		plt.xlabel("$x_1$")
		plt.ylabel("$x_2$")
		plt.title("SVM C-Support Vector Classification. $C=$" + str(Csup) )
		plt.legend(loc='best', prop={'size':9},scatterpoints=1)
		plt.grid(True)
		plt.show()
		### --------- ---------- --------- ###


	if Data2:
		### --------- Data Set 2 --------- ###
		# Load MAT file
		mat = sio.loadmat("../data/ex6data2")
		X, y = mat['X'], mat['y']
		'''
			X: (863,2)
			y: (863,1)
		'''

		# Organise data into classes
		#c1,c2 = np.where(y==1),np.where(y==0)
		c1,c2 = (y == 1).ravel(),(y == 0).ravel()

		# Train the SVM with the Radial Basis Function (RBF) 
		'''
		gamma defines how far the influence of a single training examples reaches, with low values meaning 'far' and high values meaning 'close'
		A low C-Support value makes the decision surface smooth, while a high C-Support aims to classify all training examples correctly
		'''
		Csup = 20
		gamma = 30
		clf = svm.SVC(C=Csup, kernel='rbf', gamma=gamma)
		clf.fit( X, y.flatten() )

		# Generate arrays to plot the decision boundary
		x1points,x2points,z = decision_boundary(clf, X, samples=500)

		# Plot data with decision boundary
		fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
		plt.scatter(X[c1,0], X[c1,1], marker='o',c='r',label='class 1')
		plt.scatter(X[c2,0], X[c2,1], marker='x',c='b',label='class 2')
		plt.contour(x1points, x2points, z, levels=[0], linewidth=2)
		plt.xlim([min(X[:,0]-0.1),max(X[:,0])+0.1])
		plt.ylim([min(X[:,1]-0.1),max(X[:,1])+0.1])
		plt.xlabel("$x_1$")
		plt.ylabel("$x_2$")
		plt.title("SVM Radial Basis Function. $C=$" + str(Csup) + " $\gamma =$" + str(gamma) )
		plt.legend(loc='best', prop={'size':9},scatterpoints=1)
		plt.grid(True)
		plt.show()
		### --------- ---------- --------- ###

def decision_boundary(svc, X,samples=100):
	'''
		svc is scikit-learn SVM class using C-Support Vector Classification
	'''
	x1 = np.linspace(np.min(X[:,0])-0.25,np.max(X[:,0])+0.25,samples)
	x2 = np.linspace(np.min(X[:,1])-0.25,np.max(X[:,1])+0.25,samples)

	z = np.zeros((samples,samples))
	for j in xrange(samples):
		for i in xrange(samples):
			z[j][i] = float(svc.predict([[x1[j],x2[i]]]))

	return x1,x2,z.transpose()


if __name__=='__main__':
	main()