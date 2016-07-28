# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

# Script 8: Visualisation framework

from __future__ import division

import os, math
from math import exp
from math import sqrt
import string

import numpy as np

import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from mpl_toolkits.mplot3d import Axes3D

def main():

	markers = ['d', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '+']
	# colors = ['#4B0082', 'b', '#0EBFE9', '#05E9FF', '#008000', '#99CC32', '#FF8000', '#EE0000', '#CD000CD', '#68228B', '#4B0082', 'b', '#008000', '#99CC32', '#FFE600', '#FF9912', '#EE0000', '#CD000CD', '#4B0082', 'b', '#0EBFE9', '#05E9FF', '#008000', '#99CC32', '#FF8000', '#EE0000', '#CD000CD', '#68228B', '#4B0082', 'b']
	colors = ['purple', 'b', 'aqua', 'lightseagreen', 'green', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b', 'g', 'lightgreen', 'y', 'orange', 'r', 'magenta', 'purple', 'b', 'aqua', 'lightseagreen', 'g', 'lightgreen', 'orangered', 'magenta', 'orchid', 'purple', 'darkblue', 'b']
	print len(markers)
	print len(colors)

	# x = np.array([1,2,2,3,3])
	# y = np.array([4,5,2,6,7])
	x = np.random.randint(low=0, high=4, size=30, dtype='l')
	y = np.random.randint(low=0, high=8, size=30, dtype='l')
	print x
	print y

	for m, c, i, j in zip(markers, colors, x, y):
	    plt.scatter(i, j, marker=m, c=c)

	plt.xlim(0, 4)
	plt.ylim(0, 8)

	plt.show()

def visualisation(X, X_r, color):

	fig = plt.figure()
	try:
	    ax = fig.add_subplot(211, projection='3d')
	    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
	except:
	    ax = fig.add_subplot(211)
	    ax.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)

	ax.set_title("Original data")
	ax = fig.add_subplot(212)
	ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])
	plt.title('Projected data')
	plt.show()


if __name__ == '__main__':
	main()