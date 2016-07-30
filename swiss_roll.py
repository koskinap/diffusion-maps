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
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from mpl_toolkits.mplot3d import Axes3D

datasource = './data/'


def main():

	#For testing purposes of the framework, a swiss roll dataset is generated.
	X, colors = make_swiss_roll(n_samples = 1500, noise = 0.3, random_state = None)
	
	Xlle = lle(X, colors)
	Xtsne = tsneVis(X)
	Xpca = pca_function(X)
	Xkpca = kernel_pca(X)
	Xlapl = laplacian_embedding(X)
	Xisom = isomap(X)
	Xmds = mds(X)


	# Calculate a matrix which returns pairwise distances 
	# between two datapoints according to a metric
	distMatrix = distance_matrix(X)

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
	n_components = 2
	steps = 10

	diffusionMappings = diff_mapping(s, Vh, n_components, steps)
	

	#Call Visualisation function
	visualisation(X, diffusionMappings.transpose(), Xlle, Xtsne, Xpca, Xkpca, Xlapl, Xisom, Xmds, colors)



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

	np.set_printoptions(precision = 10)

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
	# print("Computing LLE embedding")
	X_r, err = manifold.locally_linear_embedding(X, n_neighbors=20, n_components=2)
	# print("Done. Reconstruction error: %g" % err)
	return X_r

def tsneVis(data):
	tsne = manifold.TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	return tsne.fit_transform(data)

def pca_function(data):
	pca = PCA(n_components = 2, whiten = True)
	return pca.fit_transform(data)

def kernel_pca(data):
	kernelpca = KernelPCA(n_components = 2)
	return kernelpca.fit_transform(data)

def laplacian_embedding(data):
	laplacian = manifold.SpectralEmbedding(n_components = 2, affinity='nearest_neighbors')
	return laplacian.fit_transform(data)

def isomap(data):
	isom = manifold.Isomap(n_components = 2, n_neighbors=5)
	return isom.fit_transform(data)

def mds(data):
	mdsm = manifold.MDS(n_components =2)
	return mdsm.fit_transform(data)

	
def visualisation(X, XdiffMap, Xlle, Xtsne, Xpca, Xkpca, Xlapl, Xisom, Xmds, color):

	fig = plt.figure()
	try:
	    ax = fig.add_subplot(331, projection='3d')
	    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
	except:
	    ax = fig.add_subplot(331)
	    ax.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)

	ax.set_title("Original data")

	ax = fig.add_subplot(332)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on diffusion mappings space')


	ax = fig.add_subplot(333)
	ax.scatter(Xlle[:, 0], Xlle[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on lle space')

	ax = fig.add_subplot(334)
	ax.scatter(Xlle[:, 0], Xlle[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on t-SNE space')


	ax = fig.add_subplot(335)
	ax.scatter(Xpca[:, 0], Xpca[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on PCA space')

	ax = fig.add_subplot(336)
	ax.scatter(Xkpca[:, 0], Xkpca[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on kernel PCA space')

	ax = fig.add_subplot(337)
	ax.scatter(Xlapl[:, 0], Xlapl[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on laplacian embedding space')


	ax = fig.add_subplot(338)
	ax.scatter(Xisom[:, 0], Xisom[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on Isomap embedding space')

	ax = fig.add_subplot(339)
	ax.scatter(Xmds[:, 0], Xmds[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected by MDS')


	plt.xticks([]), plt.yticks([])
	plt.axis('tight')
	plt.show()
	

if __name__ == '__main__':
	main()