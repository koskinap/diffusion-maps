# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 5: Visulisation-scatterplot

import openpyxl

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import matplotlib.gridspec as gridspec

from itertools import cycle

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

datasource = './data/pcaData.xlsx'
dtsource = './data/datetimes.xlsx'

def main():
	# Read PCA data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	dtExcel = pd.ExcelFile(dtsource)

	fig = plt.figure()

	no = 331
	inc = [0,1,2,1,2,1]

	licycle = cycle(inc)
	for bactName in sheetNames:
		no += licycle.next()
		worksheet = xlData.parse(bactName)

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:1]).as_matrix()

		dtDf = dtExcel.parse(bactName)
		dt = pd.DataFrame(dtDf).as_matrix()

		scatterplot(X, bactName, no, fig, dt)

	plt.tight_layout()
	plt.show()

def  scatterplot(X, bactName, no, fig, datetimes):

	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '+']		
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen', 'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']


	ax = fig.add_subplot(no)
	ax.set_title(bactName)
	l1 = []
	names = [] # array of string

	for name_arr in datetimes.tolist():
		names.append(name_arr[0])

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
		l1.append(ax.scatter(i, j, s = 50, marker=m, c=c))

	fig.legend(l1, names, 'center right', ncol=1)

	plt.xlabel('PC1')
	plt.ylabel('PC2')

	return fig


if __name__ == '__main__':
	main()