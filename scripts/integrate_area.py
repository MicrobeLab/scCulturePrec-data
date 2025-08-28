'''
compute integrated area (Ia)

usage:
python integrate_area.py <spectrum file> <add_intensity> <out spectrum file> <resolution 1/2>
'''


import numpy as np
from scipy.integrate import simps
import sys


begin = 600
stop = 3000
#resolution = 2

def compute_integrated_area(wavenumber, intensity):
    integrated_area = simps(intensity, wavenumber)
    return integrated_area

spec_file = sys.argv[1]
add_inten = float(sys.argv[2])
out_file = sys.argv[3]
resolution = int(sys.argv[4])

fho = open(out_file, 'w')

# input: wavelength and intensity numpy arrays
w, i = np.loadtxt(spec_file, usecols=(0, 1), unpack=True)

for middle_wn in range(begin,stop+resolution,resolution):
    start_wn = middle_wn - 10
    end_wn = middle_wn + 10
    bv = (w >= start_wn) & (w <= end_wn)
    w_per = w[bv]
    i_per = i[bv] + add_inten
    ia = compute_integrated_area(w_per, i_per)
    outline = str(middle_wn) + '\t' + str(ia) + '\n'
    fho.write(outline)

fho.close()
