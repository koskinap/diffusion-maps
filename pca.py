# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 3: Principal Component Analysis

import os, math
import string
import openpyxl

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from scipy.stats import pearsonr
from numpy import corrcoef

import datetime
from itertools import cycle

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

datasource = './data/normalised/sqrtData.xlsx'
dtsource = './data/datetimes.xlsx'


def main():

	# Read data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names
	dtExcel = pd.ExcelFile(dtsource)

	# writer = pd.ExcelWriter('./data/pcaData.xlsx')
	fig = plt.figure()

	no = 331
	inc = [0,1,2,1,2,1]
	licycle = cycle(inc)
	
	for bactName in sheetNames:
		no += licycle.next()
		worksheet = xlData.parse(bactName)
		# Read time-date from file
		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()

		# Keep only the actual timeseries data, last 30 columns, and transpose
		X = pd.DataFrame(worksheet.ix[:,:29])
		XTransposed = X.as_matrix().transpose()

		# Perform PCA on Data
		pca = PCA(n_components = 2, whiten = True)
		XTransposedPca = pca.fit_transform(XTransposed)
		# scatterplot(XTransposedPca)
		scatterplot2(XTransposedPca, bactName, no, fig, dt)


		# if bactName == 'Prochlorococcus':
		# 	target = XTransposedPca
		# else:
		# 	print r2_score(y_true = target, y_pred = XTransposedPca, multioutput='raw_values')
		# 	# print r2_score(y_true = XTransposedPca.transpose(), y_pred = target, multioutput = 'variance_weighted')
		# 	# print (pearsonr(target[:,0], XTransposedPca[:,0])[1]*pearsonr(target[:,1], XTransposedPca[:,1])[1]
		# 	# corrcoef


	plt.tight_layout()
	plt.show()
	# 	pd.DataFrame(XTransposedPca).to_excel(writer, bactName)
	# writer.save()



def  scatterplot(X):

	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '+']		
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen', 'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
	    plt.scatter(i, j, s = 50, marker=m, c=c)

	plt.show()

def  scatterplot2(X, bactName, no, fig, datetimes):

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

def normalisation(dataFrame):
	return preprocessing.normalize(dataFrame)

if __name__ == '__main__':
	main()