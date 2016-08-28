# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 0: Implement diffusion framework, returns diffusion mappings

from __future__ import division

import os, math
from math import sqrt

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv

import pandas as pd

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel


def main(X, kernel = 'gaussian', n_components=2, sigma = 1, steps = 1, alpha = 0.5):

	ErrMessages = []

	if not(isinstance(n_components, int) and n_components>0):
		ErrMessages.append("Number of components must be a positive integer.")

	if not(isinstance(steps, int) and steps>0):
		ErrMessages.append("Number of steps on Markov chain must be a positive integer.")
	
	if not(sigma>0):
		ErrMessages.append("Sigma must should be a positive number.")

	if not(alpha>=0 and alpha<=1):
		ErrMessages.append("Alpha value must have a value in [0,1].")

	if not(kernel in ['gaussian','laplacian']):
		ErrMessages.append("Kernel method must be gaussian or laplacian")

	if len(ErrMessages)>0:
		for err in ErrMessages:
			print err
		exit()
		return None


	kernelMatrix = kernel_matrix(X, sigma, kernel)

	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernelMatrix, alpha)

	# Returns EVD decomposition, w are the unsorted eigenvalues, 
	# V is a matrix that in V[:,i] has the corresponding eigenvector of w[i]
	w, V = eig(probMatrix)
	# Sort eigenvectors according to decreasing order of eigenvalues 
	idx = w.argsort()[::-1]   
	eigValues = w[idx]
	eigVectors = V[:,idx]
	print eigValues[0:7]


	# Define the diffusion mapping which will be used to represent the data
	# n_components: parameter is the number of the components we need our ampping to have
	# steps: is the parameter of  the forward steps propagating on Markov process

	diffusionMappings = diff_mapping(eigValues, eigVectors, n_components, steps)
	diffusionDistances = diffusion_distance(eigValues, eigVectors, steps)

	return diffusionMappings


def kernel_matrix(X, sigma, kernel):
	print("Calculating Kernel matrix")

	# Initialise with zeros Kernel-Weight matrix
	N = X.shape[0]
	K = np.zeros((N, N))

	if kernel == 'gaussian':
		gamma = 1/(2*sigma**2)
		K = rbf_kernel(X, gamma = gamma)
	elif kernel == 'laplacian':
		gamma = 1/sigma
		K = laplacian_kernel(X, gamma = gamma)

	return K


def markov_chain(K, alpha):
	print("Calculating Markov chain matrix. \n")

	# Initialise matrices needed
	N = K.shape[0]
	d2 = np.zeros(N)
	Knorm, Dnorm, P = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))

	# Weight matrix normalisation
	d = np.sum(K, axis = 1)
	Dtemp = np.diag(d)
	np.fill_diagonal(Dnorm, 1/(Dtemp.diagonal())**alpha)
	Knorm = np.dot(Dnorm, K).dot(Dnorm)
	
	# divide normalised kernel matrix by row sum
	# convert to probability transition matrix
	d2 = np.sum(Knorm, axis = 1)
	D = np.diag(d2)

	P = np.dot(inv(D), Knorm)

	return P


def diff_mapping(eigenValues, eigenVectors, n_components , steps):
	diffMap = np.zeros((n_components, eigenVectors.shape[0]))

	# "+1" to start from right singular vector 1 and value 
	# until l+1, ignoring 0(1st singular value is equal to 1)
	for i in range(n_components):
		diffMap[i,:] = (eigenValues[i+1]**steps)*eigenVectors[:,i+1]

	return diffMap.transpose()


def diffusion_distance(eigenValues, eigenVectors, steps):
	N = eigenVectors.shape[0]
	tDiffMap = np.zeros((N-1, N))

	# "+1" to start from right eigenvector #1(not 0) 
	# until l+1, ignoring 0(1st eigenvector value is equal to 1)
	for i in range(N-1):
		tDiffMap[i,:] = (eigenValues[i+1]**steps)*eigenVectors[:,i+1]

	fullDiffMap = tDiffMap.transpose()

	diffDist = pairwise_distances(fullDiffMap, metric = 'euclidean')

	return diffDist


if __name__ == '__main__':
	main()