''' 
		K-means clustering


			- TensorFlow Example based on Jordi Torres "First contact with TensorFlow"
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

'''
	TODO:
		- Python Example
		- scikit-learn Example
'''

def main():

	''' Generate Data '''
	num_points = 2000
	vectors_set = []
	for i in xrange(num_points):
		if np.random.random() > 0.5:
			vectors_set.append([np.random.normal(0.0,0.6), np.random.normal(0.0,0.6)])
		else:
			vectors_set.append([np.random.normal(3.0,0.6), np.random.normal(1.0,0.6)])

	''' Panda Dataframe of data'''
	df = pd.DataFrame({"x": [v[0] for v in vectors_set], "y": [v[1] for v in vectors_set]})

	''' Use seaborn to plot the data in the Panda dataframe '''
	sns.lmplot("x","y",data=df,fit_reg=False,size=8)

	print "TensorFlow example"
	TensorFlow(vectors_set)

	plt.show()


def TensorFlow(vectors_set):

	vectors = tf.constant(vectors_set)
	k = 2
	centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

	expanded_vectors = tf.expand_dims(vectors,0)

	expanded_centroides = tf.expand_dims(centroides,1)

	assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors,expanded_centroides)),2),0)

	means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments,c)),[1,-1])),reduction_indices=[1]) for c in xrange(k)])

	update_centroides = tf.assign(centroides,means)

	init_op = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init_op)

	centroid_values = []
	assignment_values = []

	for step in xrange(100):
		_, centroid_values,assignment_values = sess.run([update_centroides,centroides, assignments])


	''' plot '''
	data = {"x":[],"y":[],"cluster":[]}

	for i in xrange(len(assignment_values)):
		data["x"].append(vectors_set[i][0])
		data["y"].append(vectors_set[i][1])
		data["cluster"].append(assignment_values[i])

	df = pd.DataFrame(data)
	sns.lmplot("x","y",data=df,fit_reg=False,size=8,hue="cluster",legend=False)

	

if __name__=='__main__':
	main()