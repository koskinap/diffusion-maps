# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 8: Implement diffusion framework, returns diffusion mappings of swiss roll
import numpy as np
import pandas as pd

from sklearn.datasets import make_swiss_roll
from sklearn import manifold, datasets

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D

from diffusion_framework import main as diffusion_framework

np.set_printoptions(precision=10)


def main():
	#For testing purposes of the framework, a random dataset is generated.
	dataMatrix, colors = make_swiss_roll(n_samples = 3000, noise = 0, random_state = None)

	diffusionMappings,ErrMessages = diffusion_framework(dataMatrix, kernel = 'gaussian', n_components = 4, sigma = 1, steps = 30, alpha = 0.5)
	if len(ErrMessages)>0:
		for err in ErrMessages:
			print err
		exit()


	visualisation(dataMatrix, diffusionMappings, colors)


def visualisation(X, XdiffMap, color):

	fig = plt.figure()
	try:
	    ax = fig.add_subplot(221, projection='3d')
	    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
	except:
	    ax = fig.add_subplot(221)
	    ax.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)

	ax.set_title("Original data")

	ax = fig.add_subplot(222)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 1], c=color, cmap=plt.cm.Spectral)

	plt.title('Projected data on diffusion coordinates 1-2')


	ax = fig.add_subplot(223)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 2], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on diffusion coordinates 1-3')

	ax = fig.add_subplot(224)
	ax.scatter(XdiffMap[:, 0], XdiffMap[:, 3], c=color, cmap=plt.cm.Spectral)
	plt.title('Projected data on diffusion coordinates 1-4')


	plt.xticks([]), plt.yticks([])
	plt.axis('tight')
	plt.show()


if __name__ == '__main__':
	main()