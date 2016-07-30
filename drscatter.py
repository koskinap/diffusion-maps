# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 5: Visulisation-scatterplot

import openpyxl

import numpy as np
import pandas as pd

from sklearn.utils.extmath import fast_dot
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from mpl_toolkits.mplot3d import Axes3D

from itertools import cycle

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

datasource = './data/sqrtall.xlsx'
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

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()
		# Xmds = mds(X)
		# Xle = laplacian_embedding(X)
		# Xtsne = tsneVis(X) # congested results
		Xisom = isomap(X)

		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()

		scatterplot(Xisom, bactName, no, fig, dt)

	plt.tight_layout()
	plt.show()

def  scatterplot(X, bactName, no, fig, datetimes):

	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '+']		
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen', 'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']


	ax = fig.add_subplot(no)
	ax.set_title(bactName)
	points = []
	names = []

	for name_arr in datetimes.tolist():
		names.append(name_arr[0])

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
		points.append(ax.scatter(i, j, s = 50, marker=m, c=c))

	fig.legend(points, names, 'center right', ncol=1)

	plt.xlabel('PC1')
	plt.ylabel('PC2')

	return fig

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