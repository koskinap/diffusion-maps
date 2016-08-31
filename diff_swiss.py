# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 8: Implement diffusion framework, returns diffusion mappings of swiss roll
import numpy as np
import pandas as pd

from sklearn.datasets import make_swiss_roll

from sklearn import manifold

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

from diffusion_framework import main as diffusion_framework


def main():
	#For testing purposes of the framework, a random dataset is generated.
	dataMatrix, colors = make_swiss_roll(n_samples = 1500, noise = 0, random_state = None)	

	diffusionMappings = diffusion_framework(dataMatrix, kernel = 'gaussian', \
		n_components = 4, sigma = 1, steps = 100, alpha = 0.5)

	#Set colorbar
	rng = np.arange(min(colors), max(colors))
	clrbar = cm.ScalarMappable(cmap=plt.cm.Spectral)
	clrbar.set_array(rng)

	visualisation(dataMatrix, diffusionMappings, colors, clrbar)


def visualisation(X, XdiffMap, color, clrbar):

	fig = plt.figure()

	# Plot all together
	ax = fig.add_subplot(311)
	ax.set_xlabel('Coordinate 1', fontsize = 16)
	ax.set_ylabel('Coordinate 2', fontsize = 16)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 1], c=color, cmap=plt.cm.Spectral)

	plt.title('Swiss roll on diffusion coordinates 1-2', fontsize = 18)

	ax = fig.add_subplot(312)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 2], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1', fontsize = 16)
	ax.set_ylabel('Coordinate 3', fontsize = 16)
	plt.title('Swiss roll on diffusion coordinates 1-3', fontsize = 18)

	ax = fig.add_subplot(313)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 3], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1', fontsize = 16)
	ax.set_ylabel('Coordinate 4', fontsize = 16)
	plt.title('Swiss roll on diffusion coordinates 1-4', fontsize = 18)


	fig.subplots_adjust(right=0.85)
	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	fig.colorbar(clrbar, cax=cax)
	plt.tick_params(labelsize=12)

	plt.axis('tight')
	plt.show()


	# Plot separately

	# plt.xlabel('Coordinate 1')
	# plt.ylabel('Coordinate 2')
	# plt.scatter(XdiffMap[:, 0], XdiffMap[:, 1], c=color, cmap=plt.cm.Spectral)
	# plt.show()
	# plt.clf()

	# plt.xlabel('Coordinate 1')
	# plt.ylabel('Coordinate 3')
	# plt.scatter(XdiffMap[:, 0], XdiffMap[:, 2], c=color, cmap=plt.cm.Spectral)
	# plt.show()
	# plt.clf()

	# plt.xlabel('Coordinate 1')
	# plt.ylabel('Coordinate 4')
	# plt.scatter(XdiffMap[:, 0], XdiffMap[:, 3], c=color, cmap=plt.cm.Spectral)
	# plt.show()
	# plt.clf()


if __name__ == '__main__':
	main()