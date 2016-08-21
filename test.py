
from __future__ import division

import os, math
from math import exp
from math import sqrt
import string

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



def my_dist(x):
    return np.exp(-x ** 2)

x = np.arange(0, 5)
p = my_dist(x)
print type(x)
print type(p)
plt.plot(x, p)
plt.show()

