# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 3: tSNE visualisation.

import os, math
import string
import openpyxl

import numpy as np
from numpy import linalg as LA
import pandas as pd


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import datetime

matplotlib.style.use('ggplot')

datasource = './data/Science_ortholog_cluster_counts_all_taxa.xlsx'


def main():

	# Read data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names
	
	for name in sheetNames:
		worksheet = xlData.parse(name)

		# Keep only the actual timeseries data, last 30 columns
		tsData = pd.DataFrame((normalisation(worksheet.ix[:,-30:])))

		# Transpose datamatrix
		tsDataTransposed = tsData.as_matrix().transpose()


		#Perform PCA on Data
		pca = PCA(n_components = 50, whiten = False)
		tsDataTransposedPca = pca.fit_transform(tsDataTransposed)
		
		tsneData = tsneVis(tsDataTransposedPca)

		dFrame = pd.DataFrame({'a':tsneData[:,0], 'b':tsneData[:,1]})
		scatterplot(dFrame)


		# Apply and visualise with TSNE scatterplot without PCA first
		tsneStraight = tsneVis(tsDataTransposed)
		scatterplot(pd.DataFrame({'a':tsneStraight[:,0], 'b':tsneStraight[:,1]}))

		break

def tsneVis(data):
	model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	return model.fit_transform(data)

def timeseries_plot(data):
	datetimeList = data.columns.values.tolist()
	for index, dt in enumerate(datetimeList):
		datetimeList[index] = dt.strftime('%m/%d/%Y %H:%M')

	ts = pd.Series(data.ix[1,:], index=datetimeList)
	plt.figure()
	ts.plot()
	plt.show()

def scatterplot(df):

	df.plot.scatter(x='a', y='b', color='DarkGreen')
	
	plt.show()

def normalisation(dataFrame):
	return preprocessing.normalize(dataFrame)

if __name__ == '__main__':
	main()