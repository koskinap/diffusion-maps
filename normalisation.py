# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 2.1: Normalise data in several ways and store in separate files
# In case of Sqrt normalisation, memory efficiency problems were faced, 
# so each Bacteria type was saved in separate file adn then was merged into one
# (see Script 2:sqrt_normalisation.py).

from __future__ import division

import os, math
import string
import openpyxl
from math import sqrt

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

	# Select scaling method between [MinMaxScaler, Standardisation ,Normalisation
	 # sqrt,NormalisationByRow]
	normMethod = 'CustomNormalisation'

	writer = pd.ExcelWriter('./data/normalised/' + normMethod + 'Data.xlsx')


	for bactName in sheetNames:

		worksheet = xlData.parse(bactName)
		# Keep only the actual timeseries data, last 30 columns
		X = worksheet.ix[:,-30:]
		Xdf = pd.DataFrame(X)

		# Print dimensions of working spreadsheet
		print bactName
		print worksheet.shape

		Xnp = X.as_matrix()
		# Normalise according to selected scaling method
		if normMethod == 'MinMaxScaler':
			normData = minMaxNormalisation(Xnp)

		elif normMethod == 'Normalisation':
			normData = normalisation(Xnp, 1)
		
		elif normMethod == 'Standardisation':
			normData = standarization(Xnp)

		elif normMethod == 'sqrt':
			normData = sqrt_normalisation(Xnp)

		elif normMethod == 'NormalisationByRow':
			normData = normalisation(Xnp, 0)

		elif normMethod == 'CustomNormalisation':
			normData = custom_normalisation(Xnp)
		else:
			normData = Xnp

		pd.DataFrame(normData).to_excel(writer, bactName)
		
	writer.save()

def sqrt_normalisation(data):
	N = data.shape[0]
	M = data.shape[1]
	
	norm_data = np.zeros((N,N))
	c = np.zeros(M)

	for j in range(M):
		c[j] = sum(data[:,j])
		for i in range(N):

			norm_data[i][j] = sqrt(data[i,j]/c[j])

	return norm_data


def minMaxNormalisation(data):
	
	min_max_scaler = preprocessing.MinMaxScaler()
	tsScaled = min_max_scaler.fit_transform(data)
	dfNormalized = pd.DataFrame(tsScaled)

	return dfNormalized

def normalisation(data, axis):
	return preprocessing.normalize(data, axis = axis)

def custom_normalisation(data):

	# this is close to stamdarization, no different results
	Nexp = data.shape[0]
	Ndt = data.shape[1]

	normData = np.zeros((Nexp,Ndt))
	nonZero = np.zeros(Ndt)
	mean = np.zeros(Ndt)
	std = np.zeros(Ndt)
	tempsum = np.zeros(Ndt)
	amax = np.zeros(Ndt)

	for j in range(Ndt):
		nonZero[j] = np.count_nonzero(data[:,j])
		mean[j] = sum(data[:,j])/nonZero[j]
		amax[j] = np.amax(data[:,j])
		# print amax[j]
		# print means[j]
		for i in range(Nexp):
			# Not to take into account zero elements
			if data[i,j] != 0:
				tempsum[j] += (data[i,j]-mean[j])**2

		std[j] = sqrt(tempsum[j]/nonZero[j]) 
		# print std[j]

	for i in range(Nexp):
		for j in range(Ndt):
			normData[i,j] = (data[i,j]-mean[j])/std[j]
			# normData[i,j] = (data[i,j]-mean[j])/amax[j]



	return normData

def standarization(data):
	standardScaler = preprocessing.StandardScaler()
	tsScaled = standardScaler.fit_transform(data)
	dfStandarized = pd.DataFrame(tsScaled)

	return dfStandarized

def boxplot(data):
	plt.figure()
	data.boxplot(return_type='dict')
	plt.show()

if __name__ == '__main__':
	main()