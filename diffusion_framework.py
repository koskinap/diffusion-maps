# Diffusion Maps Framework implementation as part of MSc Data Science Project of student 
# Napoleon Koskinas at University of Southampton, MSc Data Science course

import numpy as np
from numpy import linalg as LA
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import os, math
import pandas as pd
import string

datasource = './data/'



def main():
	read_data()

def read_data():
	fileList = os.listdir(datasource)
	for f in fileList:
		xlData = pd.ExcelFile(''.join([datasource, f]))

	sheetNames = xlData.sheet_names

	for name in sheetNames:
		print name
		worksheet = xlData.parse(name)
		print worksheet.head()
		break

if __name__ == '__main__':
	main()