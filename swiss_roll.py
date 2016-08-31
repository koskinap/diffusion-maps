 #  Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Scipt 4: Visualise and test diffusion maps on Swiss Roll Dataset 
# and compare withs other manifold learning techniques
import numpy as np
import pandas as pd

from sklearn.datasets import make_swiss_roll
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

from diffusion_framework import main as diffusion_framework

datasource = './data/'

def main():

	#For testing purposes of the framework, a swiss roll dataset is generated.
	X, colors = make_swiss_roll(n_samples = 2000, noise = 0.1, random_state = None)

	Xtsne = tsneVis(X)
	Xpca = pca_function(X)
	Xkpca = kernel_pca(X)
	Xmds = mds(X)

	Xlle = lle(X)
	Xlapl = laplacian_embedding(X)
	Xisom = isomap(X)
	XhesEig = hessian_eigen(X)

	#Set colorbar
	rng = np.arange(min(colors), max(colors))
	clrbar = cm.ScalarMappable(cmap=plt.cm.Spectral)
	clrbar.set_array(rng)

	#Call Visualisation functions
	visualisation(X, colors, clrbar)
	visualisation2(Xpca, Xkpca, Xmds, Xtsne, colors, clrbar)
	visualisation3(Xlle, Xlapl, Xisom, XhesEig, colors, clrbar)

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


def visualisation2(Xpca, Xkpca, Xmds, Xtsne, color, clrbar):
	fig = plt.figure()

	ax = fig.add_subplot(221)
	ax.scatter(Xpca[:, 0], Xpca[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Principal Component 1',fontsize = 16)
	ax.set_ylabel('Principal Component 2',fontsize = 16)
	plt.title('a. Projected data using PCA',fontsize = 18)

	ax = fig.add_subplot(222)
	ax.scatter(Xkpca[:, 0], Xkpca[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1',fontsize = 16)
	ax.set_ylabel('Coordinate 2',fontsize = 16)
	plt.title('b. Projected data using kernel PCA',fontsize = 18)

	ax = fig.add_subplot(223)
	ax.scatter(Xmds[:, 0], Xmds[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1',fontsize = 16)
	ax.set_ylabel('Coordinate 2',fontsize = 16)
	plt.title('c. Visualised using MDS',fontsize = 18)

	ax = fig.add_subplot(224)
	ax.scatter(Xtsne[:, 0], Xtsne[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1',fontsize = 16)
	ax.set_ylabel('Coordinate 2',fontsize = 16)
	plt.title('d. Visualised using t-SNE',fontsize = 18)


	fig.subplots_adjust(right=0.85)
	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	fig.colorbar(clrbar, cax=cax)

	plt.show()


def visualisation3(Xlle, Xlapl, Xisom, XhesEig, color, clrbar):
	fig = plt.figure()

	ax = fig.add_subplot(221)
	ax.scatter(Xlle[:, 0], Xlle[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1',fontsize = 16)
	ax.set_ylabel('Coordinate 2',fontsize = 16)
	plt.title('a. Locally Linear Embeddings',fontsize = 18)

	ax = fig.add_subplot(222)
	ax.scatter(Xlapl[:, 0], Xlapl[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1',fontsize = 16)
	ax.set_ylabel('Coordinate 2',fontsize = 16)	
	plt.title('b. Laplacian Eigenmaps',fontsize = 18)


	ax = fig.add_subplot(223)
	ax.scatter(Xisom[:, 0], Xisom[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1',fontsize = 16)
	ax.set_ylabel('Coordinate 2',fontsize = 16)
	plt.title('c. Isometric Feature Mappings',fontsize = 18)


	ax = fig.add_subplot(224)
	ax.scatter(XhesEig[:, 0], XhesEig[:, 1], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1',fontsize = 16)
	ax.set_ylabel('Coordinate 2',fontsize = 16)
	plt.title('d. Hessian Eigenmaps',fontsize = 18)

	fig.subplots_adjust(right=0.85)
	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	fig.colorbar(clrbar, cax=cax)

	plt.show()
	

def visualisation(X, color, clrbar):

	fig = plt.figure(frameon=False)

	try:
	    ax = fig.add_subplot(111, projection='3d')
	    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

	except:
	    ax = fig.add_subplot(111)
	    ax.scatter(X[:, 0],  X[:, 2], c=color, cmap=plt.cm.Spectral)

	plt.colorbar(clrbar)
	plt.axis('tight')
	plt.title('Three dimensional Swiss roll for 2000 points')

	# plt.savefig("./images/swiss.eps") # save as png
	plt.show()
	

if __name__ == '__main__':
	main()