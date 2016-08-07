# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Scipt 4: Visualise and test diffusion maps on Swiss Roll Dataset 
# and compare withs other manifold learning techniques
import numpy as np
import pandas as pd

from sklearn.datasets import make_swiss_roll
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
	X, colors = make_swiss_roll(n_samples = 1500, noise = 0, random_state = None)
	
	Xlle = lle(X, colors)
	Xtsne = tsneVis(X)
	Xpca = pca_function(X)
	Xkpca = kernel_pca(X)
	Xlapl = laplacian_embedding(X)
	Xisom = isomap(X)
	Xmds = mds(X)

	XdiffMap,ErrMessages = diffusion_framework(X, kernel = 'gaussian' , n_components = 2, sigma = 1, steps = 1, alpha = 0.5)
	if len(ErrMessages)>0:
		for err in ErrMessages:
			print err
		exit()
	
	#Call Visualisation function
	visualisation(X, XdiffMap, Xlle, Xtsne, Xpca, Xkpca, Xlapl, Xisom, Xmds, colors)


def lle(data):
	# print("Computing LLE embedding")
	X_r, err = manifold.locally_linear_embedding(data, n_neighbors=5, n_components=2)
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
	# try:
	#     ax = fig.add_subplot(331, projection='3d')
	#     ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
	# except:
	#     ax = fig.add_subplot(331)
	#     ax.scatter(X[:, 0],  X[:, 2], c=color, cmap=plt.cm.Spectral)
	ax = fig.add_subplot(331)
	ax.scatter(X[:, 0],  X[:, 2], c=color, cmap=plt.cm.Spectral)

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