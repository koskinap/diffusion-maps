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

from sklearn.metrics import r2_score
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

from diffusion_framework import main as diffusion_framework

from itertools import cycle

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

# datasource = './data/normalised/sqrtData.xlsx'
# datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/CustomNormalisationData.xlsx'
# datasource = './data/normalised/StandardisationData.xlsx'
datasource = './data/normalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/rawData.xlsx'


dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	dtExcel = pd.ExcelFile(dtsource)

	correlation = []

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		print('\nComputing embedding for:  ' + bactName)

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		# Perform dimensionality reduction, choose technique
		# drX = mds(X)
		# drX = laplacian_embedding(X)
		# drX = tsneVis(X) # congested results
		# drX = isomap(X)
		# drX = lle(X)

		drX = diffusion_framework(X, kernel = 'gaussian' , sigma = 0.5, \
		 n_components = 2, steps = 1, alpha = 0.5)
		# Read time-date from file
		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()
		# oned(drX,bactName)
		scatterbar(drX, bactName, dt)
		# onebar(drX,bactName,dt)


def oned(X,bactName):
	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', \
	'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', \
	'^', '^', '^', '^', '+']

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	fig = plt.figure()
	ax = fig.add_subplot(111)

	points = []
	
	for m, c, i in zip(markers, colors, X[:,0]):
		points.append(ax.scatter(i, 0, s = 50, marker=m, c=c))

	plt.title(bactName)
	plt.show()

def onebar(X, bactName, dt):
	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', \
	'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', \
	'^', '^', '^', '^', '+']

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.set_title(bactName)
	points = []
	names = []
	date_object = []
	times = []
	days = []

	for name_arr in dt.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for i in date_object:
		times.append(int(i.hour))
		days.append(int(i.day))

	Xx = X[:,0].tolist()
	w = (max(Xx)-min(Xx))/20

	# ax.bar(Xx, times, width=w, c=colors, alpha = 0.5)
	ax.bar(Xx, days, width=w, c=colors, alpha = 0.5)

	ax.set_xlabel('Diffusion Coordinate 1')
	ax.set_ylabel('Time')

	plt.show()

def scatterbar(X, bactName, datetimes):

	# Create a custom set of markers with custom colours to reproduce results
	# of previous research and compare after dimensionality reduction
	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', \
	'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', \
	'^', '^', '^', '^', '+']

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.set_title(bactName)
	points = []
	names = []
	date_object = []
	times = []
	days = []

	for name_arr in datetimes.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for i in date_object:
		times.append(int(i.hour))
		days.append(int(i.day)-7)


	Xx = X[:,0].tolist()
	Xy = X[:,1].tolist()
	w = (max(Xx)-min(Xx))/10
	# ax.bar(Xx, times, zs=Xy, zdir='y', width=w, color=colors, alpha = 0.7)
	ax.bar(Xx, days, zs=Xy, zdir='y', width=w, color=colors, alpha = 0.7)



	ax.set_xlabel('Diffusion Coordinate 1')
	ax.set_zlabel('Time')
	ax.set_ylabel('Diffusion coordinate 2')

	plt.show()


def lle(data):
	X_r, err = manifold.locally_linear_embedding(data, n_neighbors=5, n_components=2)
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