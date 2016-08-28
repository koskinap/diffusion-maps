# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 6: Visulisation-scatterplot of timepoints in 2D-3D space, 
# using a selected dimensionality reduction technique to find patterns

import openpyxl

import numpy as np
import pandas as pd
from math import sqrt

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA


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

# datasource = './data/normalised/rawData.xlsx'
# datasource = './data/normalised/sqrtData.xlsx'
datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/NormalisationByRowData.xlsx'
# datasource = './data/no?rmalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/MinMaxScalerFeatureData.xlsx'


dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	dtExcel = pd.ExcelFile(dtsource)

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)
		# idx = [i+1 for i in range(30)]
		idx = range(30)
		# Read time-dates from file
		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()

		print('\nComputing embedding for:  ' + bactName)

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		distanceMatrix = pairwise_distances(X, metric = 'euclidean')
		d = distanceMatrix.flatten()
		# med = np.median(d)
		# s = sqrt(med**2/1.4)

		# s = 2*np.std(d)

		# Perform dimensionality reduction, choose technique
		# drX = mds(X)
		# drX = laplacian_embedding(X)
		# drX = tsneVis(X) # congested results
		# drX = isomap(X)
		drX = lle(X)

		# drX = diffusion_framework(X, kernel = 'gaussian' , sigma = s, \
		#  n_components = 2, steps = 1, alpha = 0.5)

		#Choose visualisation
		# oned(drX,bactName)
		# scatterbar(drX, bactName, dt)
		# onebar(drX, bactName, dt, idx)
		idxbar(drX, bactName, dt, idx)
		# break


def idxbar(X, bactName, dt, idx):

	points, names, date_object, times, days, mins, hours = [], [], [], [], [], [], []

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.set_title('Diffusion maps 2nd coordinate for {}'.format(bactName), fontsize=20)


	for name_arr in dt.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for ti in date_object:
		hours.append(int(ti.hour))
		mins.append(int(ti.minute))
		times.append(ti.strftime('%H:%M'))
		days.append(int(ti.day)-6)

	Xx = X[:,0].tolist()
	Xy = X[:,1].tolist()

	# Set width of bar according to the min-max values of mapping
	# in the corresponding a coordinate
	w = 0.50
	bars = ax.bar(idx, Xx, width=w, color=colors, alpha = 0.8)
	ax.set_ylabel('Diffusion Coordinate 1', fontsize=15)

	# w = 0.50
	# bars = ax.bar(idx, Xy, width=w, color=colors, alpha = 0.8)
	# ax.set_ylabel('Diffusion Coordinate 2', fontsize=15)

	ax.set_xlabel('Time point', fontsize=15)
	ax.tick_params(labelsize=14)

	dpos = [1, 11, 19, 29]
	for dl in dpos:
		ax.axvline(x=dl, color='k', linestyle='dotted')

	plt.legend(bars, times, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='14', ncol = 1)
	# plt.savefig('./images/oned/'+bactName+'.png')
	plt.show()



def onebar(X, bactName, dt, idx):

	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.set_title(bactName)
	points, names, date_object, times, days = [], [], [], [], []


	for name_arr in dt.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for i in date_object:
		times.append(int(i.hour))
		days.append(int(i.day)-6)

	Xx = X[:,0].tolist()
	Xy = X[:,1].tolist()

	# Set width of bar according to the min-max values of mapping
	# in the corresponding a coordinate

	w = (max(Xx)-min(Xx))/50
	bars = ax.bar(Xx, days, width=w, color=colors, alpha = 0.7)
	ax.set_xlabel('Diffusion Coordinate 1', fontsize=13)


	# w = (max(Xy)-min(Xy))/50
	# bars = ax.bar(Xy, days, width=w, color=colors, alpha = 0.7)	
	# ax.set_xlabel('Diffusion Coordinate 2', fontsize=13)


	# bars = ax.bar(Xx, idx, width=w, color=colors, alpha = 0.7)
	# bars = ax.bar(Xx, times, width=w, color=colors, alpha = 0.7)
	# bars = ax.bar(Xy, idx, width=w, color=colors, alpha = 0.7)
	# bars = ax.bar(Xy, times, width=w,  color=colors, alpha = 0.7)


	# ax.set_ylabel('Time', fontsize=13)
	ax.set_ylabel('Day', fontsize=13)
	# ax.set_ylabel('Sequence number')

	plt.legend(bars, names,loc ='center left', bbox_to_anchor=(1, 0.5), fontsize='13')
	ax.tick_params(labelsize=12)

	plt.show()




def scatterbar(X, bactName, datetimes):

	# Create a custom set of markers with custom colours to reproduce results
	# of previous research and compare after dimensionality reduction
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen',\
	 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen',\
	  'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', \
	  'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.set_title(bactName)
	points, names, date_object, times, days = [], [], [], [], []

	for name_arr in datetimes.tolist():
		date_object.append(datetime.strptime(name_arr[0], '%m/%d/%Y %H:%M'))
		names.append(name_arr[0])

	for i in date_object:
		times.append(int(i.hour))
		days.append(int(i.day)-7)

	Xx = X[:,0].tolist()
	Xy = X[:,1].tolist()
	w = (max(Xx)-min(Xx))/10

	ax.bar(Xx, times, zs=Xy, zdir='y', width=w, color=colors, alpha = 0.7)
	ax.set_zlabel('Time')


	ax.set_xlabel('Diffusion Coordinate 1')
	ax.set_ylabel('Diffusion coordinate 2')

	# ax.bar(Xx, days, zs=Xy, zdir='y', width=w, color=colors, alpha = 0.7)
	# ax.set_zlabel('Day')
	# ax.set_zlabel('Sequence number')
	
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