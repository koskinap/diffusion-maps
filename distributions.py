"""
Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
Napoleon Koskinas at University of Southampton, MSc Data Science course

Script 6: Plot distribution of euclidean distances
"""

import openpyxl

import numpy as np
import pandas as pd
from math import sqrt

from sklearn.metrics.pairwise import pairwise_distances

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1



# Choose set of normalised data
# datasource = './data/normalised/rawData.xlsx'
# datasource = './data/normalised/sqrtData.xlsx'
# datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/NormalisationByRowData.xlsx'
datasource = './data/normalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/MinMaxScalerFeatureData.xlsx'


dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names
	# dtExcel = pd.ExcelFile(dtsource)

	no = 321
	fig = plt.figure()

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		# distanceMatrix = pairwise_distances(X, metric = 'sqeuclidean')
		distanceMatrix = pairwise_distances(X, metric = 'euclidean')

		d = distanceMatrix.flatten()
		histogram(d, bactName, fig, no)
		# prob_dens(d)
		no+=1
		break

	plt.show()


def histogram(d, bactName, fig, no):
	num_bins = 50
	# histogram of the data
	# n, bins, patches = plt.hist(d, num_bins, normed=1, facecolor='green', alpha=0.5)

	ax = fig.add_subplot(no)
	ax.set_title(bactName)
	n, bins, patches = ax.hist(d, num_bins, facecolor='green', alpha=0.5)

	mu = np.mean(d)
	std = np.std(d)
	md = np.median(d)


	med = ax.axvline(md, color='b', linestyle='dashed', linewidth=1)

	plt.xlabel('Euclidean distance')
	plt.ylabel('Frequency')

	plt.title('Histogram of Euclidean distances for '+ bactName )
	plt.legend([med],['Median'],loc='upper left')
	plt.title(bactName)

	plt.subplots_adjust(left=0.15)
	plt.tight_layout()
	plt.show()

def prob_dens(d):
	mu = np.mean(d)
	std = np.std(d)
	md = np.median(d)

	dmin = np.min(d)
	dmax = np.max(d)

	x = np.arange(dmin, dmax, step = 1.9)
	p = my_dist(x)

	print np.trapz(p,x)
	plt.plot(x, p)
	plt.show()

def my_dist(x):
    return np.exp(-x ** 2)
	# return np.exp(-x)


def histogram2(d):
	num_bins = 50
	# histogram of the data
	# n, bins, patches = plt.hist(d, num_bins, normed=1, facecolor='green', alpha=0.5)
	n, bins, patches = plt.hist(d, num_bins, facecolor='green', alpha=0.5)

	mu = np.mean(d)
	std = np.std(d)
	md = np.median(d)


	plt.axvline(md, color='b', linestyle='dashed', linewidth=1)

	plt.xlabel('Euclidean distance')
	plt.ylabel('Probability')
	plt.ylabel('Bin size')

	plt.title('Histogram of Euclidean distances')
	plt.subplots_adjust(left=0.15)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
