# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Scipt 4: Visualise and test diffusion maps on Swiss Roll Dataset 
# and compare withs other manifold learning techniques

from __future__ import division

import os, math
from math import exp
from math import sqrt
import string

import numpy as np
from numpy.linalg import svd

import pandas as pd

from sklearn.utils.extmath import fast_dot
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_swiss_roll
from sklearn import manifold, datasets


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from mpl_toolkits.mplot3d import Axes3D

datasource = './data/'


def main():

	#For testing purposes of the framework, a swiss roll dataset is generated.
	dataMatrix, t = make_swiss_roll(n_samples = 1500, noise = 1, random_state = None)
	# dataMatrix = np.random.rand(10,4)
	dataMatrix2 = lle(dataMatrix, t)
	# visualisation(dataMatrix, dataMatrix2, t)
	# exit()

	# Calculate a matrix which returns pairwise distances 
	# between two datapoints according to a metric
	distMatrix = distance_matrix(dataMatrix)

	# Choose a kernel function and create a kernel matrix
	# with values according to a kernel function
	kernelMatrix, totalDist = kernel_matrix(distMatrix)


	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernelMatrix, totalDist)

	# Returns SVD decomposition, s is the vector of singular values
	# U,V are expected to be square matrices
	U, s, Vh = apply_svd(probMatrix)
	# print ("Singular values")
	# print s
	# print V

	# Define the diffusion mapping which will be used to represent the data
	# n_components parameter is the number of intrinsic dimensionality of the data
	# steps is the parameter of  the forward steps propagating on Markov process

	# n_components , steps take a default value here until the algorithm is propely implemented.
	# Their values is a matter of research
	n_components = 3
	steps = 3

	diffusionMappings = diff_mapping(s, Vh, n_components, steps)
	# print diffusionMappings.shape

	visualisation(dataMatrix, diffusionMappings.transpose(), t)



def distance_matrix(dataMatrix):

	# Choose Euclidean distance
	dMatrix = pairwise_distances(dataMatrix, metric = 'euclidean')

	return dMatrix

def kernel_matrix(dMatrix):
	# Value of sigma is very important, and object of research.Here default value.
	sigma = 2

	# Define a kernel matrix

	# Get dimensions of square distance Matrix N
	N = dMatrix.shape[0]

	# Initialise with zeros Kernel matrix and distance vector
	d = np.zeros(N)
	K = np.zeros((N, N))

	# Define Gaussian kernel : exp(-(dMatrix[i, j]**2)/(2*(sigma**2)))
	for i in range(N):
		# Optimise here, exclude computation under diagonal
		for j in range(N):
			K[i, j] = exp(-(dMatrix[i, j]**2)/(2*(sigma**2)))
		d[i] = sum(K[i,:])

	return K, d


def markov_chain(K, d):
	N = K.shape[0]
	# Initialise NxN probability transition matrix
	P = np.zeros((N,N))

	# Normalise probabilities by dividing with total
	# distance to each node of the graph
	for i in range(N):
		for j in range(N):
			# P[i, j] = (K[i, j])/sqrt(d[i]**2)
			P[i, j] = K[i, j]/d[i]

	np.set_printoptions(precision=20)
	
	print ("Probability sums")
	print sum(P)

	return P

def apply_svd(P):
	# Apply SVD, return the 3 U, s, V matrices sorted by eigenvalue descending
	return svd(P, full_matrices = True, compute_uv = True)


def diff_mapping(s, V , n_components , steps):

	diffMap = np.zeros((n_components, V.shape[0]))

	# "+1" starts from column 1 until column l+1
	for i in range(n_components):
		# print (s[i+1]**t)*V[:,i+1] #wrong singular vectors
		diffMap[i,:] = (s[i+1]**steps)*V[i+1,:]

	return diffMap

def diffusion_distance(dataMatrix, diffMapMatrix):
	pass


def lle(X, color):
	print("Computing LLE embedding")
	X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=3)
	print("Done. Reconstruction error: %g" % err)
	return X_r

	
def visualisation(X, X_r, color):

	fig = plt.figure()
	try:
	    ax = fig.add_subplot(211, projection='3d')
	    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
	except:
	    ax = fig.add_subplot(211)
	    ax.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)

	ax.set_title("Original data")
	ax = fig.add_subplot(212)
	ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])
	plt.title('Projected data')
	plt.show()
	

if __name__ == '__main__':
	main()