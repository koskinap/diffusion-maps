# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 0: Working directory to test new functions. 

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
	read_data()

def read_data():

	# Read data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names
	
	for name in sheetNames:
		worksheet = xlData.parse(name)

		columns = worksheet.columns
		for col in columns:
			if (type(col)is not datetime.datetime):
				print col
				print len(worksheet[col].unique())
				# print worksheet.col.unique()


		# Keep only the actual timeseries data, last 30 columns
		tsData = worksheet.ix[:,-30:]
		# tsDataFrame = pd.DataFrame(tsData)

		# Call a function which plot time-series data
		# timeseries_plot(tsData)

		# Print dimensions of spreadsheet
		print name
		print worksheet.shape
		break

		# linegraph(tsDataFrame)

		# tSNE multidimensional visualisation


def tsneVis(data):
	pass 

def linegraph(dataFrame):

	fig = plt.figure() 
	gs = gridspec.GridSpec(dataFrame.shape[0], dataFrame.shape[1])
	for i in xrange(dataFrame.shape[0]):
		for j in xrange(dataFrame.shape[1]):
			ax = plt.subplot(dataFrame[i+1,j+1])	
			ax.set_ylabel('Foo') #Add y-axis label 'Foo' to graph 'ax' (xlabel for x-axis)
			fig.add_subplot(ax) #add 'ax' to figure
	
	fig.show()
	
def scatterplot():
	pass
	
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