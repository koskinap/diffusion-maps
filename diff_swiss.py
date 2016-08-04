# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 8: Implement diffusion framework, returns diffusion mappings of swiss roll

from __future__ import division

import os, math
from math import exp
from math import sqrt
import string
import csv

import numpy as np
from numpy.linalg import svd
from numpy.linalg import eig
from numpy.linalg import inv

import pandas as pd

from sklearn.utils.extmath import fast_dot
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_swiss_roll
from sklearn import manifold, datasets

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from mpl_toolkits.mplot3d import Axes3D


np.set_printoptions(precision=10)


def main():
	#For testing purposes of the framework, a random dataset is generated.
	dataMatrix, colors = make_swiss_roll(n_samples = 3000, noise = 0.1, random_state = None)

	# Calculate a matrix which returns pairwise distances 
	# between two datapoints according to a metric
	distMatrix = distance_matrix(dataMatrix)

	# Choose a kernel function and create a kernelMatrixernel matrix
	# with values according to a kernel function
	kernelMatrix, totalDist = kernel_matrix(distMatrix)

	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernelMatrix, totalDist)

	# Returns SVD decomposition, s is the vector of singular values
	# U,V are expected to be square matrices
	U, s, Vh = apply_svd(probMatrix)

	# Define the diffusion mapping which will be used to represent the data
	# n_components: parameter is the number of intrinsic dimensionality of the data
	# steps: is the parameter of  the forward steps propagating on Markov process

	# n_components , steps take a default value here until the algorithm is propely implemented.
	# Their values is a matter of research
	n_components = 4
	steps = 1

	diffusionMappings = diff_mapping(s, Vh, n_components, steps)
	visualisation(dataMatrix, diffusionMappings, colors)



def distance_matrix(dataMatrix):
	print("Calculating distance matrix \n")

	# Choose Euclidean distance
	dMatrix = pairwise_distances(dataMatrix, metric = 'euclidean')

	return dMatrix

def kernel_matrix(dMatrix):

	print("Calculating Kernel matrix \n")

	# Value of sigma is very important, and objective of research.Here default value.
	sigma = 1

	# Define a kernel matrix

	# Get dimensions of square distance Matrix N
	N = dMatrix.shape[0]

	# Initialise with zeros Kernel matrix and distance vector
	d = np.zeros(N)
	K = np.zeros((N, N))

	# Define Gaussian kernel : exp(-(dMatrix[i, j]**2)/(2*(sigma**2)))
	for i in range(N):
		for j in range(N):
			K[i, j] = exp(-(dMatrix[i, j]**2)/(2*(sigma**2)))
		# d[i] = sum(K[i,:])

	d = np.sum(K, axis = 1)
	# print d

	return K, d


def markov_chain(K, d):
	print("Calculating Markov chain matrix \n")

	N = K.shape[0]
	# Initialise NxN probability transition matrix
	P = np.zeros((N,N))

	# Normalise distances by converting to probabilities by 
	# dividing with total distance to each node of the graph

	# Solution 1
	d2 = np.zeros(N)
	Knorm = np.zeros((N,N))

	for i in range(N):
		for j in range(N):
			Knorm[i, j] = K[i, j]/(sqrt(d[i])*sqrt(d[j]))
		d2[i] = sum(Knorm[i,:])

	# Convert to probability matrix by normalising by sum of row, 
	# which makes the probability of every row of matrix P equal to 1 
	D = np.diag(d2)
	P = np.inner(inv(D), Knorm)

	
	# print ("\n Probability sums \n")
	# for i in range(N):
	# 	print sum(P[i,:])

	# P is not symmetric!!!
	return P

def apply_svd(P):
	print('Computing SVD \n')
	# Apply SVD, return the 3 U, s, Vh matrices sorted by singular value descending
	return svd(P, full_matrices = True, compute_uv = True)


def diff_mapping(s, Vh , n_components , steps):
	diffMap = np.zeros((n_components, Vh.shape[0]))

	# "+1" starts from column 1 until column l+1
	for i in range(n_components):
		diffMap[i,:] = (s[i+1]**steps)*Vh[i+1,:]

	return diffMap.transpose()

def visualisation(X, XdiffMap, color):

	fig = plt.figure()
	try:
	    ax = fig.add_subplot(221, projection='3d')
	    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
	except:
	    ax = fig.add_subplot(221)
	    ax.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)

	ax.set_title("Original data")

	ax = fig.add_subplot(222)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 1], c=color, cmap=plt.cm.Spectral)

	plt.title('Projected data on diffusion mappings space, 1-2')


	ax = fig.add_subplot(223)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 2], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on diffusion space 1-3')

	ax = fig.add_subplot(224)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 3], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on diffusion space 1-4')


	plt.xticks([]), plt.yticks([])
	plt.axis('tight')
	plt.show()


if __name__ == '__main__':
	main()