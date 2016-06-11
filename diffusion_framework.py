# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

import os, math
import string
# from libc.math import sqrt, exp

import numpy as np
from numpy import linalg as LA
import pandas as pd

from sklearn.utils.extmath import fast_dot
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


datasource = './data/'


def main():
	dataMatrix = np.zeros((3, 3))
	print dataMatrix
	exit()

	# Calculate a matrix which returns pairwise distances 
	# between two datapoints according to a metric
	distMatrix = distance_matrix(dataMatrix)

	# Choose a kernel function and create a kernel matrix
	# with values according to a kernel function
	kernelMatrix, totalDist = kernel_matrix(dMatrix)

	# Create probability transition matrix from kernel matrix and total dist.
	probMatrix = markov_chain(kernel_matrix,totalDist)

def distance_matrix(dataMatrix):

	# Choose Euclidean distance
	dMatrix = pairwise_distances(dataMatrix, metric = distance_metrics[2])

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
			K[i, j] = exp(- (dMatrix[i, j]**2)/(2*sigma**2))
			d[i] += K[i,j]

	return K, d


def markov_chain(K, d):
	N = K.shape[0]
	# Initialise NxN probability transition matrix
	P = np.zeros((N,N))

	for i in range(N):
		for j in range(N):
			P[i, j] = K[i, j]/d[i]

	# from future import division???

	return P

def svd():
	pass

def mapping():
	pass

if __name__ == '__main__':
	main()