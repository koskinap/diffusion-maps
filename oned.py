"""Diffusion Maps implementation as part of MSc Data Science Project of student 
Napoleon Koskinas at University of Southampton, MSc Data Science course

Script 6: Visulisation-scatterplot of timepoints in 2D-3D space, 
using a selected dimensionality reduction technique to find patterns
"""

import openpyxl

import numpy as np
import pandas as pd
from math import sqrt

from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

from diffusion_framework import main as diffusion_framework


matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1


# Select datasource
# datasource = './data/normalised/rawData.xlsx'
# datasource = './data/normalised/sqrtData.xlsx'
datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/NormalisationByRowData.xlsx'
# datasource = './data/normalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/MinMaxScalerFeatureData.xlsx'


dtsource = './data/datetimes.xlsx'

def main():

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	dtExcel = pd.ExcelFile(dtsource)

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)
		idx = range(30)
		# Read time-dates from file
		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()

		print('\nComputing embedding for:  ' + bactName)

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		# Compute distance matrix to extract information
		distanceMatrix = pairwise_distances(X, metric = 'euclidean')
		d = distanceMatrix.flatten()
		
		# Select sigma by median
		# med = np.median(d)
		# s = sqrt(med**2/1.4)

		# Select distance matrix std as sigma
		s = np.std(d)

		# Perform dimensionality reduction, choose technique.
		# Diffusion maps default choice

		# drX = mds(X)
		# drX = laplacian_embedding(X)
		# drX = tsneVis(X) # congested results
		# drX = isomap(X)
		# drX = lle(X)

		drX = diffusion_framework(X, kernel = 'gaussian' , sigma = s, \
		 n_components = 2, steps = 1, alpha = 0.5)

		# Choose visualisation technique, description provided below
		scatterbar(drX, bactName, dt, colors)
		onebar(drX, bactName, dt, idx, colors)
		idxbar(drX, bactName, dt, idx,colors)
		break


"""
Creates a barplot by representing in form of 
a bar the diffusion coordinate in a chronological index order
"""
def idxbar(X, bactName, dt, idx, colors):

	points, names, date_object, times, days, mins, hours = [], [], [], [], [], [], []

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# Get time objects for later use
	for name_arr in dt.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for ti in date_object:
		hours.append(int(ti.hour))
		mins.append(int(ti.minute))
		times.append(ti.strftime('%H:%M'))
		days.append(int(ti.day)-6)

	# Get diffusion coordinates
	Xx = X[:,0].tolist()
	Xy = X[:,1].tolist()

	# Set width of bar according to the min-max values of mapping
	# in the corresponding a coordinate. Choose coordinate accordingly
	w = 0.50
	bars = ax.bar(idx, Xx, width=w, color=colors, alpha = 0.8)
	ax.set_ylabel('Diffusion Coordinate 1', fontsize=15)
	ax.set_title('Diffusion maps 1st coordinate for {}'.format(bactName), fontsize=20)


	# w = 0.50
	# bars = ax.bar(idx, Xy, width=w, color=colors, alpha = 0.8)
	# ax.set_ylabel('Diffusion Coordinate 2', fontsize=15)
	# ax.set_title('Diffusion maps 2nd coordinate for {}'.format(bactName), fontsize=20)


	ax.set_xlabel('Time point', fontsize=15)
	ax.tick_params(labelsize=14)

	dpos = [1, 11, 19, 29]
	for dl in dpos:
		ax.axvline(x=dl, color='k', linestyle='dotted')

	plt.legend(bars, times, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='14', ncol = 1)
	plt.show()


"""
Creates a barplot by representing in form of 
a bar a day of the time in xaxis the samples are set by coordinate.
Opposite functionality to idxbar
"""
def onebar(X, bactName, dt, idx, colors):

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# ax.set_title(bactName)
	points, names, date_object, times, days = [], [], [], [], []

	# Get time objects for later use
	for name_arr in dt.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for i in date_object:
		times.append(int(i.hour))
		days.append(int(i.day)-6)

	# Get diffusion coordinates
	Xx = X[:,0].tolist()
	Xy = X[:,1].tolist()

	# Set width of bar according to the min-max values of mapping
	# in the corresponding a coordinate. Choose coordinate according to preference
	w = (max(Xx)-min(Xx))/50
	bars = ax.bar(Xx, days, width=w, color=colors, alpha = 0.7)
	ax.set_xlabel('Diffusion Coordinate 1', fontsize=13)

	# w = (max(Xy)-min(Xy))/50
	# bars = ax.bar(Xy, days, width=w, color=colors, alpha = 0.7)	
	# ax.set_xlabel('Diffusion Coordinate 2', fontsize=13)

	ax.set_ylabel('Day', fontsize=13)


	#Choose one out of two to print time to the corresponding coordinate
	# bars = ax.bar(Xx, times, width=w, color=colors, alpha = 0.7)
	# bars = ax.bar(Xy, times, width=w,  color=colors, alpha = 0.7)
	# ax.set_ylabel('Time', fontsize=13)


	#Choose one out of two to print time to the corresponding coordinate
	# bars = ax.bar(Xx, idx, width=w, color=colors, alpha = 0.7)
	# bars = ax.bar(Xy, idx, width=w, color=colors, alpha = 0.7)
	# ax.set_ylabel('Index', fontsize=13)

	plt.legend(bars, names,loc ='center left', bbox_to_anchor=(1, 0.5), fontsize='13')
	ax.tick_params(labelsize=12)

	plt.show()



"""
Creates a 3D barplot by representing in form of 
a bar a day of the time/day in zaxis while the samples are scattered in two-dimensional
space by two first coordinates.
"""
def scatterbar(X, bactName, datetimes, colors):

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.set_title(bactName)
	points, names, date_object, times, days = [], [], [], [], []

	# Get time objects for later use
	for name_arr in datetimes.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for i in date_object:
		times.append(int(i.hour))
		days.append(int(i.day)-7)

	# Get diffusion coordinates
	Xx = X[:,0].tolist()
	Xy = X[:,1].tolist()

	# Set width of bar according to the min-max values of mapping
	# in the corresponding a coordinate. Choose coordinate according to preference
	w = (max(Xx)-min(Xx))/10
	ax.bar(Xx, times, zs=Xy, zdir='y', width=w, color=colors, alpha = 0.7)
	ax.set_zlabel('Time')

	ax.set_xlabel('Diffusion Coordinate 1')
	ax.set_ylabel('Diffusion coordinate 2')

	# Uncomment to make bar length show the day
	# ax.bar(Xx, days, zs=Xy, zdir='y', width=w, color=colors, alpha = 0.7)
	# ax.set_zlabel('Day')
	
	plt.show()


"""
Alternative manifold learning techniques
"""

def lle(data):
	X_r, err = manifold.locally_linear_embedding(data, n_neighbors=10, n_components=2)
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
	isom = manifold.Isomap(n_components = 2, n_neighbors=10)
	return isom.fit_transform(data)

if __name__ == '__main__':
	main()