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

import pandas as pd

from sklearn.utils.extmath import fast_dot
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_swiss_roll

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


datasource = './data/'


def main():

	#For testing purposes of the framework, a swiss roll dataset is generated.
	# dataMatrix, t = make_swiss_roll(n_samples = 4, noise = 0.05, random_state = None)
	dataMatrix = np.random.rand(10,2)
	# dataMatrix = np.ones((10, 5))

	# Calculate a matrix which returns pairwise distances 
	# between two datapoints according to a metric
	distMatrix = distance_matrix(dataMatrix)

	# Choose a kernel function and create a kernel matrix
	# with values according to a kernel function
	kernelMatrix, totalDist = kernel_matrix(distMatrix)
	# print("kernelMatrix")
	# print kernelMatrix

	# print("totalDist")
	# print totalDist



	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernelMatrix, totalDist)

	# print("probMatrix")
	# print probMatrix
	# exit()

	# Returns SVD decomposition, s is the vector of singular values
	# U,V are expected to be square matrices
	U, s, V = apply_svd(probMatrix)
	# print ("Singular values")
	# print s

	# Define the diffusion mapping which will be used to represent the data
	# n_components: parameter is the number of intrinsic dimensionality of the data
	# steps: is the parameter of  the forward steps propagating on Markov process

	# n_components , steps take a default value here until the algorithm is propely implemented.
	# Their values is a matter of research
	n_components = 2
	steps = 3

	diffusionMappings = diff_mapping(s, V, n_components, steps)
	# print diffusionMappings.shape



def distance_matrix(dataMatrix):

	# Choose Euclidean distance
	dMatrix = pairwise_distances(dataMatrix, metric = 'euclidean')

	return dMatrix

def kernel_matrix(dMatrix):
	# Value of sigma is very important, and objective of research.Here default value.
	sigma = 0.00000030

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

	# Normalise distances by converting to probabilities by 
	# dividing with total distance to each node of the graph
	for i in range(N):
		for j in range(N):
			# P[i, j] = (K[i, j])/sqrt(d[i]**2)
			P[i, j] = K[i, j]/d[i]

	np.set_printoptions(precision=20)
	
	print ("Probability sums")
	print np.sum(P, axis = 1)

	# P is not symmetric!!!
	return P

def apply_svd(P):
	# Apply SVD, return the 3 U, s, Vh matrices sorted by singular value descending
	return svd(P, full_matrices = True, compute_uv = True)


def diff_mapping(s, V , n_components , steps):
	diffMap = np.zeros((n_components, V.shape[0]))

	# "+1" starts from column 1 until column l+1
	for i in range(n_components):
		diffMap[i,:] = (s[i+1]**steps)*V[i+1,:]

	return diffMap

def diffusion_distance(dataMatrix, diffMapMatrix):
	# May not be needed
	pass
	

if __name__ == '__main__':
	main()