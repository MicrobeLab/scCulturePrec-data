'''
Raman Spectral Preprocessing

Before using, please download the Singularity image and Shared Object File from https://figshare.com/s/25c804e477f855e66cc8

Usage:

singularity exec raman_preprocessing.sif python raman_preprocessing.py --input /path/to/raw_spect.txt --output /path/to/preprocessed_spect.txt

'''

import sys
sys.path.append('/path/to/.so')  # change to path containing 'RamanPreprocessing.cpython-36m-x86_64-linux-gnu.so'
import numpy as np
from RamanPreprocessing import CRR
from RamanPreprocessing import airPLS
from RamanPreprocessing import MaxMinNormalization
from scipy.signal import savgol_filter
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, help='/path/to/input_spec.txt')
	parser.add_argument('--output', type=str, help='/path/to/output_spec.txt')
	parser.add_argument('--start', type=int, help='start wavenumber (default=400 cm-1)', default=400)
	parser.add_argument('--end', type=int, help='end wavenumber (default=3600 cm-1)', default=3600)
	parser.add_argument('--silent_start', type=int, help='start wavenumber for region without signal (default=1800 cm-1)', default=1800)
	parser.add_argument('--silent_end', type=int, help='end wavenumber for region without signal (default=2000 cm-1)', default=2000)
	parser.add_argument('--snr_min', type=float, help='minimum SNR (default=2.0)', default=2.0)
	parser.add_argument('--crr_fs', type=int, help='CRR filter size (default=9)', default=9)
	parser.add_argument('--crr_df', type=float, help='CRR dynamic factor (default=3)', default=3)
	parser.add_argument('--win_sg', type=int, help='window size for SG (default=5)', default=5)
	parser.add_argument('--n_sg', type=int, help='order of polynomial for SG (default=3)', default=3)
	parser.add_argument('--airpls_lamb', type=int, help='lambda for airPLS (default=100)', default=100)
	parser.add_argument('--airpls_iter', type=int, help='itermax for airPLS (default=15)', default=15)
	parser.add_argument('--fix_norm', action='store_true', help='normalize using min at silent region')

	args = parser.parse_args()

	# load a spectrum
	spec = np.loadtxt(args.input)
	# keep spectral intensity between 400 and 3600 cm-1
	spec = spec[(spec[:, 0] >= args.start) & (spec[:, 0] <= args.end)]
	# cosmic ray removal
	intensity = spec[:, 1]
	intensity = CRR(intensity, args.crr_fs, args.crr_df)
	# noise reduction using the Savitzky-Golay filter
	intensity = savgol_filter(intensity, args.win_sg, args.n_sg)
	# airPLS baseline correction
	intensity = airPLS(intensity, args.airpls_lamb, args.airpls_iter)
	intensity = np.array(intensity)
	# normalization
	if args.fix_norm:
		peak = max(intensity) 
		silent_spec = intensity[(spec[:, 0] >= args.silent_start) & (spec[:, 0] <= args.silent_end)]
		silent_med = np.median(silent_spec)
		intensity = (intensity - silent_med) / (peak - silent_med)
	else:
		intensity = MaxMinNormalization(intensity)
	# snr filtering
	signal_range = max(intensity) - min(intensity)
	silent_spec = intensity[(spec[:, 0] >= args.silent_start) & (spec[:, 0] <= args.silent_end)]
	noise_range = max(silent_spec) - min(silent_spec)
	snr = signal_range / noise_range
	if snr > args.snr_min:
		spec[:, 1] = intensity
		np.savetxt(args.output, spec, fmt='%f', delimiter="\t")
	else:
		print(f'{args.input}: SNR is {snr}, SNR < {args.snr_min}, skipped')


if __name__ == "__main__":
	main()
