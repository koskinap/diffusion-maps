# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

import os, math
from math import exp
import string
# from libc.math import sqrt, exp

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

	dataMatrix, t = make_swiss_roll(n_samples=100, noise=0.05, random_state=None)

	# Calculate a matrix which returns pairwise distances 
	# between two datapoints according to a metric
	distMatrix = distance_matrix(dataMatrix)


	# Choose a kernel function and create a kernel matrix
	# with values according to a kernel function
	kernelMatrix, totalDist = kernel_matrix(distMatrix)

	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernelMatrix,totalDist)

	#proof that probabilities sum to 1
	# for p in probMatrix:
	# 	print sum(p)
	U, S, V = apply_svd(probMatrix)
	print S


def distance_matrix(dataMatrix):

	# Choose Euclidean distance
	dMatrix = pairwise_distances(dataMatrix, metric = 'euclidean')

	return dMatrix

def kernel_matrix(dMatrix):
	# Value of sigma is very important, and objective of research.Here default value.
	sigma = 0.3

	# Define a kernel matrix

	# Get dimensions of square distance Matrix N
	N = dMatrix.shape[0]

	# Initialise with zeros Kernel matrix and distance vector
	d = np.zeros(N)
	K = np.zeros((N, N))

	# Define Gaussian kernel : exp(-dMatrix(i,j)/(2*sigma**2))
	for i in range(N):
		# Optimise here, exclude computation under diagonal
		for j in range(N):
			K[i, j] = exp(-(dMatrix[i, j]**2)/(2*sigma**2))
			d[i] += K[i,j]

	return K, d


def markov_chain(K, d):
	N = K.shape[0]
	# Initialise NxN probability transition matrix
	P = np.zeros((N,N))

	# Normalise probabilities by dividing with total
	# distance to each node of the graph
	for i in range(N):
		for j in range(N):
			P[i, j] = K[i, j]/d[i]

	return P

def apply_svd(P):
	# Apply SVD, return the 3 U, S, V matrices sorted by eigenvalue desc
	return svd(P, full_matrices = True, compute_uv = True)

def mapping():
	pass

if __name__ == '__main__':
	main()