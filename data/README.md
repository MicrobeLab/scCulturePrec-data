# Morphological and Spectral Data

- All the morphological and spectral data are available on [figshare](https://doi.org/10.6084/m9.figshare.29993929). The morphological data are also provided in the `morphol` directory above. 

- The columns in the morphological data tables, in order, are: area, width, length, aspect ratio, eccentricity, circularity, convexity, extent, perimeter, centroid. 

- The spectral data tables included integrated area (±10 cm<sup>-1</sup>) from 600 cm<sup>-1</sup> to 3000 cm<sup>-1</sup>.

## Data Format
Data are provided as NumPy arrays, with each array containing samples from one taxon, shaped as `[number_of_samples, number_of_features]`. The [`scCulturePrec`](https://github.com/MicrobeLab/scCulturePrec) Python package can be used to load and format the data for model training.
