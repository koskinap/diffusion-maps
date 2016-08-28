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

np.set_printoptions(precision=5)


def main():
	#For testing purposes of the framework, a random dataset is generated.
	dataMatrix, colors = make_swiss_roll(n_samples = 1500, noise = 0.1, random_state = None)	

	diffusionMappings = diffusion_framework(dataMatrix, kernel = 'gaussian', \
		n_components = 4, sigma = 1, steps = 1, alpha = 0.5)

	visualisation(dataMatrix, diffusionMappings, colors)


def visualisation(X, XdiffMap, color):

	fig = plt.figure()

	# Plot all together
	ax = fig.add_subplot(311)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 2')
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 1], c=color, cmap=plt.cm.Spectral)

	plt.title('Projected data on diffusion coordinates 1-2')

	ax = fig.add_subplot(312)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 2], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 3')
	# plt.title('Projected data on diffusion coordinates 1-3')

	ax = fig.add_subplot(313)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 3], c=color, cmap=plt.cm.Spectral)
	ax.set_xlabel('Coordinate 1')
	ax.set_ylabel('Coordinate 4')
	# plt.title('Projected data on diffusion coordinates 1-4')

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