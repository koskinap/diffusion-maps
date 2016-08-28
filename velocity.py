# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 9: Network example taken on 20/08/2016 from 
# https://networkx.readthedocs.io/en/stable/examples/drawing/weighted_graph.html

import openpyxl

import numpy as np
import pandas as pd
from math import sqrt

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from diffusion_framework import main as diffusion_framework

matplotlib.style.use('ggplot')

# Choose set of normalised data
# datasource = './data/normalised/sqrtData.xlsx'
# datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/CustomNormalisationData.xlsx'
# datasource = './data/normalised/StandardisationData.xlsx'
# datasource = './data/normalised/MinMaxScalerData.xlsx'
datasource = './data/normalised/MinMaxScalerFeatureData.xlsx'
# datasource = './data/normalised/rawData.xlsx'
# datasource = './data/pcaData.xlsx'

dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	dtExcel = pd.ExcelFile(dtsource)

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		dtDf = dtExcel.parse(bactName)
		dtMatrix = pd.DataFrame(dtDf).as_matrix()

		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()
		print 2*np.std(X)
		drX = diffusion_framework(X, kernel = 'gaussian' , sigma = 2*np.std(X), \
		 n_components = 2, steps = 1, alpha = 0.5)
		
		# drX = pd.DataFrame(worksheet).as_matrix()

		N = drX.shape[0]
		V = np.zeros((N-2, drX.shape[1]))
		idx = [i+1 for i in range(N-2)]


		times = []
		for time_arr in dtMatrix.tolist():
			times.append(datetime.strptime(time_arr[0], '%m/%d/%Y %H:%M'))

		dt = []
		for j in idx:
			next = j+1
			prev = j-1
			dt.append((times[next]-times[prev]).total_seconds())


		for j in idx:
			next = j+1
			prev = j-1
			# V[prev,:] = (drX[next,:] - drX[prev,:])/2
			V[prev,:] = (drX[next,:] - drX[prev,:])/dt[prev]

		plt.plot(V[:,0], drX[1:29,0])
		plt.show()


		break


if __name__ == '__main__':
	main()
