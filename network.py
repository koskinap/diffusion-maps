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
import networkx as nx

matplotlib.style.use('ggplot')

# Choose set of normalised data
# datasource = './data/normalised/rawData.xlsx'
datasource = './data/normalised/sqrtData.xlsx'
# datasource = './data/normalised/NormalisationData.xlsx'
# datasource = './data/normalised/NormalisationByRowData.xlsx'
# datasource = './data/normalised/MinMaxScalerData.xlsx'
# datasource = './data/normalised/MinMaxScalerFeatureData.xlsx'


dtsource = './data/datetimes.xlsx'

def main():

	xlData = pd.ExcelFile(datasource)
	sheetNames = xlData.sheet_names

	# dtExcel = pd.ExcelFile(dtsource)

	for bactName in sheetNames:
		worksheet = xlData.parse(bactName)

		X = pd.DataFrame(worksheet.ix[:,:29]).as_matrix().transpose()

		sigma = 23.57
		g = 1/sigma**2
		K = rbf_kernel(X, gamma = g)

		# Initialise graph and add weights
		G=nx.Graph()
		for i in range(K.shape[0]):
			for j in range(K.shape[1]):
				G.add_edge(str(i),str(j), weight=K[i,j])


		estrong=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
		eweak=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]

		print len(estrong)
		print len(eweak)

		pos=nx.spring_layout(G) # positions for all nodes

		# nodes
		nx.draw_networkx_nodes(G,pos,node_size=200)

		# edges
		nx.draw_networkx_edges(G,pos, edgelist=estrong, width=1)
		nx.draw_networkx_edges(G,pos, edgelist=eweak, width=1,alpha=0.5,edge_color='b',style='dashed')

		# labels
		nx.draw_networkx_labels(G,pos,font_size=15,font_family='sans-serif')

		plt.axis('off')
		# plt.savefig("weighted_graph.png") # save as png
		plt.show() # display

		break

if __name__ == '__main__':
	main()

