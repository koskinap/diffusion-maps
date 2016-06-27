# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

import os, math
import string
import openpyxl

import numpy as np
from numpy import linalg as LA
import pandas as pd


import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn import preprocessing

import datetime

matplotlib.style.use('ggplot')


datasource = './data/Science_ortholog_cluster_counts_all_taxa.xlsx'


def main():
	read_data()

def read_data():

	# Read data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	writer = pd.ExcelWriter('./data/summary.xlsx')
	
	for name in sheetNames:
		worksheet = xlData.parse(name)
		# Keep only the actual timeseries data, last 30 columns
		tsData = worksheet.ix[:,-30:]

		#minmaxscaler
		normalised_data = data_normalisation(tsData)
		# sklearn_normalize
		normalised_data2 = normalisation2(tsData)
		standarized_data = standarization(tsData)

		# Call a function which plot time-series data
		# timeseries_plot(ts_data)

		# Print dimensions of spreadsheet
		print name
		print worksheet.shape
		
		# Create a boxplot per timestamp per bacteria
		# boxplot(ts_data)
		boxplot(normalised_data)
		boxplot(normalised_data2)
		boxplot(standarized_data)
		
		# Return summary statistics
		describe(tsData, name, writer)
		
	writer.save()

		

def describe(data, bactName, writer):
	ts = pd.DataFrame(data)
	ts.describe(include = 'all').to_excel(writer, bactName)


def boxplot(data):
	ts = pd.DataFrame(data)

	plt.figure()
	ts.boxplot(return_type='dict')
	plt.show()

def scatterplot():
	pass

def data_normalisation(data):
	ts = pd.DataFrame(data)
	
	min_max_scaler = preprocessing.MinMaxScaler()
	ts_scaled = min_max_scaler.fit_transform(ts)
	df_normalized = pd.DataFrame(ts_scaled)

	return df_normalized

def normalisation2(data):
	ts = pd.DataFrame(data)
	return preprocessing.normalize(ts)

def standarization(data):
	ts = pd.DataFrame(data)

	standard_scaler = preprocessing.StandardScaler()
	ts_scaled = standard_scaler.fit_transform(ts)
	df_standarized = pd.DataFrame(ts_scaled)

	return df_standarized
	
def timeseries_plot(data):
	datetimeList = data.columns.values.tolist()
	for index, dt in enumerate(datetimeList):
		datetimeList[index] = dt.strftime('%m/%d/%Y %H:%M')

	# print data.ix[1,:]
	# exit()
	ts = pd.Series(data.ix[1,:], index=datetimeList)
	# ts = pd.DataFrame(data.ix[0:1,:], index=datetimeList)
	plt.figure()
	ts.plot()
	plt.show()



if __name__ == '__main__':
	main()