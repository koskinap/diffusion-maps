import os
import openpyxl

import numpy as np
import pandas as pd

# Script 1.1: Gets date-time points for future use(mainly in graphs)

# This script gathers the timedates of the timeseries datapoints 
# for future use e.g. as legends in figures

datasource = './data/Science_ortholog_cluster_counts_all_taxa.xlsx'

def main():
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	writer = pd.ExcelWriter('./data/datetimes.xlsx')

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)
		# Keep only the actual timeseries data, last 30 columns
		X = worksheet.ix[:,-30:]
		# Gets names of columns, which are time-date points
		datetimeList = list(X.columns.values)
		for index, dt in enumerate(datetimeList):
			datetimeList[index] = dt.strftime('%m/%d/%Y %H:%M')

		pd.DataFrame(datetimeList).to_excel(writer, bactName)

	writer.save()
		

if __name__ == '__main__':
	main()