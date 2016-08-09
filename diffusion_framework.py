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
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_swiss_roll

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


datasource = './data/'
np.set_printoptions(precision=10)

def main(X, kernel = 'gaussian', n_components =2, sigma = 1, steps = 1, alpha = 0.5):

	ErrMessages = []

	if not(isinstance(n_components, int) and n_components>0):
		ErrMessages.append("Number of components should be a positive integer.")

	if not(isinstance(steps, int) and steps>0):
		ErrMessages.append("Number of steps on Markov chain should be a positive integer.")
	
	if not(sigma>0):
		ErrMessages.append("Sigma value should be a positive number.")

	if not(alpha>=0 and alpha<=1):
		ErrMessages.append("Alpha value should have a value in [0,1].")

	if not(kernel in ['gaussian','cosine', 'polynomial']):
		ErrMessages.append("Kernel method should be gaussian, polynomial or cosine")

	if len(ErrMessages)>0:
		return None, ErrMessages


	kernelMatrix = kernel_matrix(X, sigma, kernel)

	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernelMatrix, alpha)

	# Returns SVD decomposition, s is the vector of singular values
	# U,V are expected to be square matrices
	U, s, V = apply_svd(probMatrix)
	print("Singular values")
	print s[1:]

	# Define the diffusion mapping which will be used to represent the data
	# n_components: parameter is the number of the components we need our ampping to have
	# steps: is the parameter of  the forward steps propagating on Markov process
	# Their values is a matter of research

	diffusionMappings = diff_mapping(s, V, n_components, steps)
	diffusionDistances = diffusion_distance(s, V, steps)

	return diffusionMappings, ErrMessages


def kernel_matrix(X, sigma, kernel):

	print("Calculating Kernel matrix")

	# Value of sigma is very important, and objective of research.Here default value.
	# Get dimensions of square distance Matrix N
	N = X.shape[0]
	# Initialise with zeros Kernel matrix
	K = np.zeros((N, N))

	if kernel == 'gaussian':
		gamma = 0.5/sigma**2
		K = rbf_kernel(X, gamma = gamma)
	elif kernel=='cosine':
		K = pairwise_distances(X, metric = kernel)

	return K


def markov_chain(K, alpha):
	print("Calculating Markov chain matrix. \n")

	# Initialise NxN probability transition matrix
	N = K.shape[0]
	P = np.zeros((N,N))
	d = np.sum(K, axis = 1)

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

	#Solution 2 according to van Maaten, without probability matrix, on symmetric
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

	# "+1" to start from right singular vector 1 and value 
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