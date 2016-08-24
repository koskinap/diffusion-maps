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


dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	# dtExcel = pd.ExcelFile(dtsource)

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()
		drX = diffusion_framework(X, kernel = 'gaussian' , sigma = 23.5, \
		 n_components = 2, steps = 1, alpha = 0.5)
