# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 0: Implement diffusion framework, returns diffusion mappings

from __future__ import division

import os, math
from math import exp
from math import sqrt
import string

import numpy as np
from numpy.linalg import svd
from numpy.linalg import inv
from scipy.linalg import svd as scipysvd

import pandas as pd

from sklearn.utils.extmath import fast_dot
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_swiss_roll

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


datasource = './data/'
np.set_printoptions(precision=10)

def main(X,  n_components, sigma, steps, alpha):

	ErrMessages = []
	if not(isinstance(n_components, int) and n_components>0):
		ErrMessages.append("Number of components should be a positive integer.")

	if not(isinstance(steps, int) and steps>0):
		ErrMessages.append("Number of steps on Markov chain should be a positive integer.")
	
	if not(sigma>0):
		ErrMessages.append("Sigma value should be a positive number.")

	if not(alpha>=0 and alpha<=1):
		ErrMessages.append("Alpha value should have a value in [0,1].")


	if len(ErrMessages)>0:
		return None, ErrMessages

	# For testing purposes of the framework, a swiss roll dataset is generated.
	# dataMatrix = np.random.rand(100,2)
	dataMatrix = X

	# Calculate a matrix which returns pairwise distances 
	# between two datapoints according to a metric
	distMatrix = distance_matrix(dataMatrix)

	# Choose a kernel function and create a kernel matrix
	# with values according to a kernel function
	kernelMatrix, totalDist = kernel_matrix(distMatrix, sigma)

	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernelMatrix, totalDist, alpha)

	# Returns SVD decomposition, s is the vector of singular values
	# U,V are expected to be square matrices
	U, s, V = apply_svd(probMatrix)
	# print("Singular values")
	# print s

	# Define the diffusion mapping which will be used to represent the data
	# n_components: parameter is the number of the components we need our ampping to have
	# steps: is the parameter of  the forward steps propagating on Markov process
	# Their values is a matter of research

	diffusionMappings = diff_mapping(s, V, n_components, steps)
	diffusionDistances = diffusion_distance(s, V, steps)
	print diffusionDistances.shape
	return diffusionMappings, ErrMessages

def distance_matrix(dataMatrix):
	print("Calculating distance matrix")

	# Choose Euclidean distance
	dMatrix = pairwise_distances(dataMatrix, metric = 'euclidean')

	return dMatrix

def kernel_matrix(dMatrix, sigma):

	print("Calculating Kernel matrix")

	# Value of sigma is very important, and objective of research.Here default value.

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

	d = np.sum(K, axis = 1)

	return K, d


def markov_chain(K, d, alpha):
	print("Calculating Markov chain matrix")

	N = K.shape[0]
	# Initialise NxN probability transition matrix
	P = np.zeros((N,N))

	# Normalise distances by converting to probabilities by 
	# dividing with total distance to each node of the graph
	d2 = np.zeros(N)
	Knorm = np.zeros((N,N))
	Dnorm = np.zeros((N,N))
	
	Dtemp = np.diag(d)
	np.fill_diagonal(Dnorm, 1/(Dtemp.diagonal()**alpha))
	Knorm = np.dot(Dnorm, K).dot(Dnorm)

	# Solution 1
	d2 = np.sum(Knorm, axis = 1)
	D = np.diag(d2)
	P = np.dot(inv(D), Knorm)


	#Solution 2 according to van Maaten, without probability matrix
	# d2 = np.sum(Knorm, axis = 1)
	# d3 = np.sqrt(d2)
	# D = np.dot(d3[:,None],d3.transpose()[None,:])
	# P = np.divide(Knorm, D)

	# Make sure total sum of each row is equal to 1
	# print ("\n Probability sums \n")
	# for i in range(N):
	# 	print sum(P[:,i])
	# P is not symmetric!!!
	return P

def apply_svd(P):
	# Apply SVD, return the 3 U, s, Vh matrices sorted by singular value descending
	return svd(P, full_matrices = True, compute_uv = True)
	# return scipysvd(P, full_matrices = True, computes_uv = True)

def diff_mapping(s, V , n_components , steps):
	diffMap = np.zeros((n_components, V.shape[0]))

	# "+1" to start from right singular vector 1 
	# until l+1, ignoring 0(1st singular value is equal to 1)
	for i in range(n_components):
		diffMap[i,:] = (s[i+1]**steps)*V[i+1,:]

	return diffMap.transpose()

def diffusion_distance(s, V, steps):
	N = V.shape[0]
	tDiffMap = np.zeros((N-1, N))

	# "+1" to start from right singular vector 1 
	# until l+1, ignoring 0(1st singular value is equal to 1)
	for i in range(N-1):
		tDiffMap[i,:] = (s[i+1]**steps)*V[i+1,:]
	
	fullDiffMap = tDiffMap.transpose()

	diffDist = pairwise_distances(fullDiffMap, metric = 'euclidean')

	return diffDist


if __name__ == '__main__':
	main()