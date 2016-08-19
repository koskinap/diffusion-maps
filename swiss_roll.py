 #  Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Scipt 4: Visualise and test diffusion maps on Swiss Roll Dataset 
# and compare withs other manifold learning techniques
import numpy as np
import pandas as pd

from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_s_curve
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

from diffusion_framework import main as diffusion_framework

datasource = './data/'


def main():

	#For testing purposes of the framework, a swiss roll dataset is generated.
	X, colors = make_swiss_roll(n_samples = 1500, noise = 0.1, random_state = None)
	# X, colors = make_s_curve(n_samples = 1500, noise = 0.1, random_state = None)


	Xlle = lle(X)
	Xlapl = laplacian_embedding(X)
	Xisom = isomap(X)
	XhesEig = hessian_eigen(X)

	Xtsne = tsneVis(X)
	Xpca = pca_function(X)
	Xkpca = kernel_pca(X)
	Xmds = mds(X)


	# XdiffMap,ErrMessages = diffusion_framework(X, kernel = 'gaussian' , n_components = 2, sigma = 1, steps = 1, alpha = 0.5)
	# if len(ErrMessages)>0:
	# 	for err in ErrMessages:
	# 		print err
	# 	exit()
	
	#Call Visualisation functions
	visualisation(X, colors)
	visualisation2(Xpca, Xkpca, Xmds, Xtsne, colors)
	visualisation3(Xlle, Xlapl, Xisom, XhesEig, colors)



def lle(data):
	Xemb, err = manifold.locally_linear_embedding(data, n_neighbors=12, n_components=2)
	return Xemb

def hessian_eigen(data):
	XhesEig, err = manifold.locally_linear_embedding(data, method = 'hessian' , n_neighbors=12, n_components=2)
	return XhesEig

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


def visualisation2(Xpca, Xkpca, Xmds, Xtsne, color):
	fig = plt.figure()

	ax = fig.add_subplot(221)
	ax.scatter(Xpca[:, 0], Xpca[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Principal Component 1')
	ax.set_ylabel('Principal Component 2')
	plt.title('a. Projected data using PCA')

	ax = fig.add_subplot(222)
	ax.scatter(Xkpca[:, 0], Xkpca[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')
	plt.title('b. Projected data using kernel PCA')

	ax = fig.add_subplot(223)
	ax.scatter(Xmds[:, 0], Xmds[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')
	plt.title('c. Visualised using MDS')

	ax = fig.add_subplot(224)
	ax.scatter(Xtsne[:, 0], Xtsne[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')
	plt.title('d. Visualised using t-SNE')

	plt.axis('tight')
	plt.show()


def visualisation3(Xlle, Xlapl, Xisom, XhesEig,  color):
	fig = plt.figure()

	ax = fig.add_subplot(221)
	ax.scatter(Xlle[:, 0], Xlle[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')
	plt.title('a. Local Linear Embeddings')

	ax = fig.add_subplot(222)
	ax.scatter(Xlapl[:, 0], Xlapl[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')	
	plt.title('b. Laplacian Eigenmappings')


	ax = fig.add_subplot(223)
	ax.scatter(Xisom[:, 0], Xisom[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')
	plt.title('c. Isometric Feature Mappings')


	ax = fig.add_subplot(224)
	ax.scatter(XhesEig[:, 0], XhesEig[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')
	plt.title('d. Hessian Eigenmappings')

	plt.axis('tight')
	plt.show()
	

def visualisation(X, color):

	fig = plt.figure()
	
	try:
	    ax = fig.add_subplot(111, projection='3d')
	    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
	except:
	    ax = fig.add_subplot(111)
	    ax.scatter(X[:, 0],  X[:, 2], c=color, cmap=plt.cm.Spectral)

	plt.axis('tight')
	plt.show()
	

if __name__ == '__main__':
	main()