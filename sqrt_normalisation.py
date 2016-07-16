# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 4: Normalise data in the same way it was normalised in previous research
# and store it in an Excel file

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

	writer = pd.ExcelWriter('./data/sqrt_raw_data.xlsx')


	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)
		# Keep only the actual timeseries data, last 30 columns
		tsData = worksheet.ix[:,-30:]
		tsDataFrame = pd.DataFrame(tsData)

		# Print dimensions of working spreadsheet
		print bactName
		print worksheet.shape

		normData = sqrt_normalisation(tsData.as_matrix())
		pd.DataFrame(normData).to_excel(writer, bactName)
		break


		# break
	writer.save()




def sqrt_normalisation(data):
	N = data.shape[0]
	M = data.shape[1]
	
	norm_data = np.zeros((N,N))
	c = np.zeros(M)

	for j in range(M):
		# print data[i,j]
		c[j] = sum(data[:,j])
		for i in range(N):
			# print data[i,j]

			norm_data[i][j] = sqrt(data[i,j]/c[j])

	return norm_data


def boxplot(data):
	plt.figure()
	data.boxplot(return_type='dict')
	plt.show()


if __name__ == '__main__':
	main()