'''
python interpolation.py [input_spec.txt] [output_spec.txt]

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

import sys

w, i = np.loadtxt(sys.argv[1], usecols=(0, 1), unpack=True)
sp = UnivariateSpline(w, i, k=3, s=0)  
x = np.arange(w[0], w[-1]+1, 1.0)
y = sp(x)
xy = np.column_stack((x, y))

np.savetxt(sys.argv[2], xy, fmt='%f', delimiter="\t")
