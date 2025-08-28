import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='/path/to/input_spec.txt (two column format)')
parser.add_argument('--ref', type=str, help='/path/to/ref_spec.txt (one ref material per column)')
parser.add_argument('--plot', action='store_true', help='plot original vs fit spectrum')
parser.add_argument('--quant_out', type=str, help='/path/to/quantification_output.txt)', default=None)
parser.add_argument('--start_wn', type=int, help='start wn of cell spectrum to cut', default=None)
parser.add_argument('--end_wn', type=int, help='end wn of cell spectrum to cut', default=None)


args = parser.parse_args()
inp = args.input
ref = args.ref
plot = args.plot
quant_out = args.quant_out
start_wn = args.start_wn
end_wn = args.end_wn

reference_spectra = np.loadtxt(ref)

complex_spectrum = np.loadtxt(inp)
if start_wn is not None and end_wn is not None:
	complex_spectrum = complex_spectrum[(complex_spectrum[:, 0] >= start_wn) & (complex_spectrum[:, 0] <= end_wn)]
complex_spectrum = complex_spectrum[:, 1]


# Linear combination fitting using non-negative least squares (NNLS)
coefficients, residual_norm = nnls(reference_spectra, complex_spectrum)

fitted_spectrum = np.dot(coefficients, reference_spectra.T)

# Plotting the results
if plot:
	plt.figure(figsize=(12, 8))

	plt.subplot(3, 1, 1)
	plt.plot(complex_spectrum, label='Complex Spectrum')
	plt.plot(fitted_spectrum, label='Fitted Spectrum', linestyle='--')
	plt.legend()
	plt.title('Complex Spectrum and Fitted Spectrum')
	plt.show()

# Print the coefficients and relative amounts
	print("Coefficients:", coefficients)

# output
if quant_out is not None:
    with open(quant_out, 'w') as fh:
        for i in coefficients:
            fh.write(str(i) + '\n')




