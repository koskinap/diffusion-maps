# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 2: Create boxplots of original and normalised/standarized data

import os, math
import string
import openpyxl

import numpy as np
from numpy import linalg as LA
import pandas as pd


import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import preprocessing

import datetime

matplotlib.style.use('ggplot')


datasource = './data/Science_ortholog_cluster_counts_all_taxa.xlsx'


def main():

	# Read data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)
		# Keep only the actual timeseries data, last 30 columns
		X = worksheet.ix[:,-30:]
		Xdf = pd.DataFrame(X)
		# X2df = sqrtXldata.parse(bactName)

		# Print dimensions of working spreadsheet
		print bactName
		print worksheet.shape

		#minmaxscaler --> returns dataframe
		minMaxScaledX = minMaxNormalisation(Xdf)
		
		# sklearn_normalize --> returns ndarray
		normalisedX = normalisation(X)
		normalisedXdf = pd.DataFrame(X)
		
		#sklearn-standarize --> returns dataframe
		standarizedX = standarization(Xdf)
		
		# Create a boxplot per timestamp per bacteria
		boxplot(X, 'Original data')
		boxplot(minMaxScaledX, 'Data normalised with MinMaxScaler')
		boxplot(normalisedXdf, 'Normalised data')
		boxplot(standarizedX, 'Standarised data')
		# boxplot(X2df, 'Square data')
		break


		


def minMaxNormalisation(dataFrame):
	
	min_max_scaler = preprocessing.MinMaxScaler()
	tsScaled = min_max_scaler.fit_transform(dataFrame)
	dfNormalized = pd.DataFrame(tsScaled)

	return dfNormalized

def normalisation(dataFrame):
	return preprocessing.normalize(dataFrame)

def standarization(dataFrame):
	standardScaler = preprocessing.StandardScaler()
	tsScaled = standardScaler.fit_transform(dataFrame)
	dfStandarized = pd.DataFrame(tsScaled)

	return dfStandarized

def boxplot(data, title):
	plt.figure()
	data.boxplot(return_type='dict')
	plt.show()


if __name__ == '__main__':
	main()