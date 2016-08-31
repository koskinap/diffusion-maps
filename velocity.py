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
from sklearn.decomposition import PCA

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from diffusion_framework import main as diffusion_framework

matplotlib.style.use('ggplot')

# Choose set of normalised data
# datasource = './data/normalised/sqrtData.xlsx'
# datasource = './data/normalised/NormalisationData.xlsx'
datasource = './data/normalised/NormalisationByRowData.xlsx'
# datasource = './data/normalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/MinMaxScalerFeatureData.xlsx'
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
		distanceMatrix = pairwise_distances(X, metric = 'euclidean')
		d = distanceMatrix.flatten()
		s= np.std(d)

		dX = diffusion_framework(X, kernel = 'gaussian' , sigma = s, \
		 n_components = 5, steps = 1, alpha = 0.5)
		
		pca = PCA(n_components = 2, whiten = True)
		drX = pca.fit_transform(dX)

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

		plt.plot(V[:,1], drX[1:29,1])
		plt.show()

		# break


if __name__ == '__main__':
	main()
