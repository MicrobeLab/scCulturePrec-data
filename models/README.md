# Pre-trained Model Weights

The pre-trained model weights for the deep neural network and elastic net models are available on [figshare](https://doi.org/10.6084/m9.figshare.30000025). Pre-computed reference features for each species are also available on [figshare](https://doi.org/10.6084/m9.figshare.30001300).

## Model Card

### Model description

**Architecture:** A Siamese network (FT-Transformer for morphological features + ResNet for Raman spectra) extracts feature vectors from single-cell data. An elastic net classifier then predicts taxon membership based on distances between feature vectors and reference library entries.

**Model variants:** Species-level and genus-level classifiers using morphology only, Raman spectra only, or both data types combined.

### Training data
25 bacterial species (14 genera) from human microbiome; ~500 cells per species; 10 morphological features + Raman spectra (600–3000 cm⁻¹).

### Hardware & software
1 × NVIDIA A100 GPU (40 GB VRAM), 8 CPU cores; PyTorch, scikit-learn.
