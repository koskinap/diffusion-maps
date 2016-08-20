# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 6: Visulisation-scatterplot of timepoints in 2D space, 
# using a selected dimensionality reduction technique

import openpyxl

import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import gaussian_kde


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import networkx as nx


from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')
matplotlib.rcParams['legend.scatterpoints'] = 1

# Choose set of normalised data
# datasource = './data/normalised/sqrtData.xlsx'
# datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/CustomNormalisationData.xlsx'
# datasource = './data/normalised/StandardisationData.xlsx'
datasource = './data/normalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/rawData.xlsx'


dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	# dtExcel = pd.ExcelFile(dtsource)

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()
		sigma  = 5
		g = 3

		K = rbf_kernel(X, gamma = 0.5)

		G=nx.Graph()
		for i in range(K.shape[0]):
			for j in range(K.shape[1]):
				G.add_edge(str(i),str(j), weight=K[i,j])

		elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
		esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]

		pos=nx.spring_layout(G) # positions for all nodes

		# nodes
		nx.draw_networkx_nodes(G,pos,node_size=500)

		# edges
		nx.draw_networkx_edges(G,pos, edgelist=elarge, width=2)
		nx.draw_networkx_edges(G,pos, edgelist=esmall, width=2,alpha=0.5,edge_color='b',style='dashed')

		# labels
		nx.draw_networkx_labels(G,pos,font_size=15,font_family='sans-serif')

		plt.axis('off')
		# plt.savefig("weighted_graph.png") # save as png
		plt.show() # display

		break

def my_dist(x):
    return np.exp(-x ** 2)




if __name__ == '__main__':
	main()

