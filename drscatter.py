# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 6: Visulisation-scatterplot of timepoints in 2D space, 
# using a selected dimensionality reduction technique

import openpyxl

import numpy as np
import pandas as pd

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

from diffusion_framework import main as diffusion_framework

from itertools import cycle

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

datasource = './data/normalised/sqrtData.xlsx'
# datasource = './data/normalised/NormalisationData.xlsx'

dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	dtExcel = pd.ExcelFile(dtsource)

	fig = plt.figure()

	no = 331
	inc = [0,1,2,1,2,1]

	licycle = cycle(inc)
	for bactName in sheetNames:
		no += licycle.next()
		worksheet = xlData.parse(bactName)

		print('\nComputing embedding for')
		print bactName

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		# Perform dimensionality reduction, cho
		# drX = mds(X)
		# drX = laplacian_embedding(X)
		# drX = tsneVis(X) # congested results
		# drX = isomap(X)
		# drX = lle(X)
		drX, ErrMessages = diffusion_framework(X, kernel = 'gaussian' , n_components = 2, sigma = 0.4, steps = 1, alpha = 0.5)
		if len(ErrMessages)>0:
			for err in ErrMessages:
				print err
			exit()
		# Read time-date from file
		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()
		scatterplot(drX, bactName, no, fig, dt)

	plt.tight_layout()
	plt.show()

def  scatterplot(X, bactName, no, fig, datetimes):

	# Create a custom set of markers with custom colours to reproduce results
	 # of previous research and compare after dimensionality reduction
	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', \
	'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', \
	'^', '^', '^', '^', '+']

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	ax = fig.add_subplot(no)
	ax.set_title(bactName)
	points = []
	names = []

	for name_arr in datetimes.tolist():
		names.append(name_arr[0])

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
		points.append(ax.scatter(i, j, s = 50, marker=m, c=c))

	fig.legend(points, names, 'center right', ncol=1)

	plt.xlabel('Component 1')
	plt.ylabel('Component 2')

	return fig


def lle(data):
	# print("Computing LLE embedding")
	X_r, err = manifold.locally_linear_embedding(data, n_neighbors=5, n_components=2)
	# print("Done. Reconstruction error: %g" % err)
	return X_r

def laplacian_embedding(data):
	laplacian = manifold.SpectralEmbedding(n_components = 2, affinity='nearest_neighbors')
	return laplacian.fit_transform(data)

def mds(data):
	mdsm = manifold.MDS(n_components =2)
	return mdsm.fit_transform(data)

def tsneVis(data):
	model = manifold.TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	return model.fit_transform(data)

def isomap(data):
	isom = manifold.Isomap(n_components = 2, n_neighbors=5)
	return isom.fit_transform(data)

if __name__ == '__main__':
	main()