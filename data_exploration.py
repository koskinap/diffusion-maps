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
		ts_data = worksheet.ix[:,-30:]

		# Call a function which plot time-series data
		# timeseries_plot(ts_data)

		# Print dimensions of spreadsheet
		print worksheet.shape
		
		# Create a boxplot per timestamp ???
		# boxplot(ts_data)
		
		# Return summary statistics
		describe(ts_data, name, writer)
		
	writer.save()

		

def describe(data, bactName, writer):

	ts = pd.DataFrame(data)
	ts.describe(include = 'all').to_excel(writer, bactName)


def boxplot(data):

	ts = pd.DataFrame(data)

	plt.figure()
	ts.boxplot(return_type='dict')
	plt.show()


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