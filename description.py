# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

#Script 1 : Get summary statistics about each species of bacteria

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

	writer = pd.ExcelWriter('./data/summary.xlsx')

	for name in sheetNames:
		worksheet = xlData.parse(name)
		# Keep only the actual timeseries data, last 30 columns
		tsData = worksheet.ix[:,-30:]
		tsDataFrame = pd.DataFrame(tsData)

			# Print dimensions of spreadsheet
		print name
		print worksheet.shape

		describe(tsData, name, writer)
		
	writer.save()


def describe(data, bactName, writer):
	ts = pd.DataFrame(data)
	ts.describe(include = 'all').to_excel(writer, bactName)


if __name__ == '__main__':
	main()