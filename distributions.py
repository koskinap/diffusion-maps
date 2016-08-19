# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 6: Visulisation-scatterplot of timepoints in 2D space, 
# using a selected dimensionality reduction technique

import openpyxl

import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import gaussian_kde


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

# Choose set of normalised data
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

	# dtExcel = pd.ExcelFile(dtsource)

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		# distanceMatrix = pairwise_distances(X, metric = 'sqeuclidean')
		distanceMatrix = pairwise_distances(X, metric = 'euclidean')


		d = distanceMatrix.flatten()

		mu = np.mean(d)
		std = np.std(d)
		md = np.median(d)

		# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
		# density = gaussian_kde(d)
		# xs = np.linspace(0,10,900)
		# density.covariance_factor = lambda : .25
		# density._compute_covariance()
		# plt.plot(xs,density(xs))
		# plt.show()


		num_bins = 20
		# the histogram of the data
		n, bins, patches = plt.hist(d, num_bins, normed=1, facecolor='green', alpha=0.5)
		# add a 'best fit' line
		y = mlab.normpdf(bins, mu, std)
		plt.axvline(md, color='b', linestyle='dashed', linewidth=2)
		plt.plot(bins, y, 'r')
		plt.xlabel('Euclidean distance')
		plt.ylabel('Probability')
		plt.title('Histogram of Euclidean distances')

		# Tweak spacing to prevent clipping of ylabel
		plt.subplots_adjust(left=0.15)
		plt.show()
		break

		# print sqrt(2)*np.median(distanceMatrix)
		# sqrt(2)*np.median(distanceMatrix)

def my_dist(x):
    return np.exp(-x ** 2)




if __name__ == '__main__':
	main()
