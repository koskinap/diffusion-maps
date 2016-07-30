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

import datetime

matplotlib.style.use('ggplot')

datasource = './data/sqrtall.xlsx'

def main():

	# Read data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	# writer = pd.ExcelWriter('./data/pcaData.xlsx')

	
	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		# Keep only the actual timeseries data, last 30 columns
		# tsData = pd.DataFrame((normalisation(worksheet.ix[:,-30:])))
		# X = pd.DataFrame((normalisation(worksheet.ix[:,:29])))
		X = pd.DataFrame(worksheet.ix[:,:29])


		# Transpose datamatrix
		XTransposed = X.as_matrix().transpose()

		# Perform PCA on Data
		pca = PCA(n_components = 2, whiten = True)
		XTransposedPca = pca.fit_transform(XTransposed)
		scatterplot(XTransposedPca)

	# 	pd.DataFrame(XTransposedPca).to_excel(writer, bactName)

	# writer.save()



def  scatterplot(X):

	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '+']		
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen', 'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
	    plt.scatter(i, j, marker=m, c=c)

	plt.show()

def normalisation(dataFrame):
	return preprocessing.normalize(dataFrame)

if __name__ == '__main__':
	main()