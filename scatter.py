# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 6: Visulisation-scatterplot of timepoints in 2D space, 
# using a selected dimensionality reduction technique

import openpyxl

import numpy as np
import pandas as pd

from math import sqrt

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from diffusion_framework import main as diffusion_framework

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

# Choose set of normalised data
# datasource = './data/normalised/rawData.xlsx'
# datasource = './data/normalised/sqrtData.xlsx'
datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/NormalisationByRowData.xlsx'
# datasource = './data/normalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/MinMaxScalerFeatureData.xlsx'


dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	# sheetNames = xlData.sheet_names

	# sheetNames = ['Prochlorococcus', 'Roseobacter', 'SAR324']
	sheetNames = ['SAR11']

	dtExcel = pd.ExcelFile(dtsource)


	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)
		print('\nComputing embedding for:  ' + bactName)

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		# Perform dimensionality reduction, choose technique

		slist =	np.arange(1, 30, 0.1)
		# for s in slist:
		# 	print('s='+str(s))
		# 	drX = diffusion_framework(X, kernel = 'gaussian' , sigma = s,\
		# 	 n_components = 2, steps = 1, alpha = 0.5)

		s=3.7
		drX = diffusion_framework(X, kernel = 'gaussian' , sigma = s,\
			 n_components = 2, steps = 1, alpha = 0.5)

		# Read time-date from file
		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()

		scatterplot(drX, bactName, dt)

		break


def  scatterplot(X, bactName, datetimes):

	# Create a custom set of markers with custom colours to reproduce results
	 # of previous research and compare after dimensionality reduction
	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', \
	'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', \
	'^', '^', '^', '^', '+']

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	# fig = plt.figure()
	# fig.set_title(bactName)
	points = []
	names = []

	for name_arr in datetimes.tolist():
		names.append(name_arr[0])

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
		points.append(plt.scatter(i, j, s = 50, marker=m, c=c))

	# plt.legend(points, names, 'lower center', ncol=5, fontsize=13)
	plt.tick_params(labelsize=12)

	plt.xlabel('Diffusion coordinate 1', fontsize=14)
	plt.ylabel('Diffusion coordinate 2', fontsize=14)

	plt.show()


def lle(data):
	X_r, err = manifold.locally_linear_embedding(data, n_neighbors=3, n_components=2)
	return X_r

def laplacian_embedding(data):
	laplacian = manifold.SpectralEmbedding(n_components = 2, affinity='nearest_neighbors')
	return laplacian.fit_transform(data)

def mds(data):
	mdsm = manifold.MDS(n_components =2)
	return mdsm.fit_transform(data)

def tsneVis(data):
	model = manifold.TSNE(n_components=2, perplexity=5, early_exaggeration=20, random_state=0)
	np.set_printoptions(suppress=True)
	return model.fit_transform(data)

def isomap(data):
	isom = manifold.Isomap(n_components = 2, n_neighbors=5)
	return isom.fit_transform(data)

if __name__ == '__main__':
	main()