# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 7: Create boxplots of original and normalised/standarized data

import openpyxl

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import datetime
from itertools import cycle


matplotlib.style.use('ggplot')

originalDataSource = './data/Science_ortholog_cluster_counts_all_taxa.xlsx'

datasource = './data/normalised/'
filesource = ['RawData.xlsx','NormalisationData.xlsx','StandardisationData.xlsx', \
'MinMaxScalerData.xlsx', 'sqrtData.xlsx','NormalisationByRowData.xlsx','MinMaxScalerFeatureData.xlsx']
dtsource = './data/datetimes.xlsx'

titles = ['Original data', 'Normalised data', 'Standardised data', \
'Scaled data with MinMaxScaler','Square root divided by sum data','Feature normalisation','Scaled data with MinMaxScaler by feature']

def main():

	# Read data, get names of the 6 sheets, for different types of bacteria
	bactData = pd.ExcelFile( datasource + filesource[0])
	sheetNames = bactData.sheet_names

	# Load datetime points from file
	dtArray = pd.DataFrame(pd.ExcelFile(dtsource).parse()).as_matrix().tolist()
	dtList = ['']
	for dt in dtArray:
		dtList.append(dt[0])

	for bactName in sheetNames:

		for index, f in enumerate(filesource):

			xlData = pd.ExcelFile(datasource + f)
			print f
			worksheet = xlData.parse(bactName)
			# Keep only the actual timeseries data, last 30 columns
			X = worksheet.ix[:,:29]

			# Print dimensions of working spreadsheet
			print bactName
			print worksheet.shape

			X.boxplot(return_type='dict')

			plt.xlabel('Time point')
			plt.xticks(range(31), dtList, rotation='vertical')

			plt.ylabel('Values')

			# plt.title(titles[index])
			plt.tight_layout()
			plt.savefig("./images/"+titles[index]+bactName+".eps")
			plt.clf()

		if bactName == 'SAR11':
			break
		# break
		# break

def boxplot(data, fig, title, no, dtList):

	ax = fig.add_subplot(no)
	ax.set_title(title)

	data.boxplot(return_type='dict')

	plt.xlabel('Time point')
	plt.ylabel('Values')

	return fig


if __name__ == '__main__':
	main()