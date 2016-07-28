# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 5: Visulisation-scatterplot

import openpyxl

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

matplotlib.style.use('ggplot')

datasource = './data/pcaData.xlsx'

def main():
	# Read PCA data, get names of the 6 sheets, for different types of bacteria
	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	fig = plt.figure()

	no = 321
	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		# Keep only the actual timeseries data, last 30 columns
		X = pd.DataFrame(worksheet.ix[:,:1]).as_matrix()

		scatterplot(X, bactName, no, fig)
		no += 1


	plt.tight_layout()
	plt.show()

def  scatterplot(X, bactName, no, fig):

	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '+']		
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen', 'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	ax = fig.add_subplot(no)
	ax.set_title(bactName)

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
		ax.scatter(i, j, s = 50, marker=m, c=c)
	plt.xlabel('PC1')
	plt.ylabel('PC2')

	return fig

def  scatterplot2(X):

	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '+']		
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen', 'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']

	for m, c, i, j in zip(markers, colors, X[:,0], X[:,1]):
	    plt.scatter(i, j, marker=m, c=c)

	plt.show()


if __name__ == '__main__':
	main()