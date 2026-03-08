# scCulturePrec-data

Supporting data and scripts for "Precision culturomics enabled by unlabeled single-cell morphology and Raman spectra".



------------------------------------------------------------------------



## Bacterial Single-Cell Morphology Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/) [![Segment Anything](https://img.shields.io/badge/SAM-Meta%20AI-purple.svg)](https://github.com/facebookresearch/segment-anything)

Scripts for extracting bacterial single-cell morphological features from microscopy images. The pipeline combines computer vision techniques with Meta AI's Segment Anything Model (SAM) to achieve cell segmentation and quantitative morphology analysis.

### Overview of the workflow

1.  **Box Detection** - Identify cell regions using either automated or manual methods
2.  **Segmentation** - Generate cell masks using SAM with box prompts
3.  **Morphological Analysis** - Extract quantitative morphological features

Example input and output files are available [here](https://github.com/MicrobeLab/scCulturePrec-data/tree/main/morphology_analysis).

### Box Detection Modes

| Mode          | Script          | Best For                  | Description                                                                                      |
|------------------|------------------|------------------|-------------------|
| **Automatic** | `box_auto.py`   | High-throughput screening | Computer vision pipeline with adaptive thresholding, CLAHE enhancement, and watershed separation |
| **Manual**    | `box_manual.py` | Complex/irregular cells   | Interactive PyGUI for manual selection and annotation                                            |

### Morphological Features (10 Metrics)

| Feature           | Unit | Description                                   |
|-------------------|------|-----------------------------------------------|
| **Area**          | μm²  | Cell surface area                             |
| **Width**         | μm   | Short axis of minimum area rectangle          |
| **Length**        | μm   | Long axis of minimum area rectangle           |
| **Aspect Ratio**  | \-   | Length/Width ratio                            |
| **Eccentricity**  | \-   | √(1 - (minor²/major²)) from fitted ellipse    |
| **Circularity**   | \-   | 4π × Area / Perimeter² (1.0 = perfect circle) |
| **Extent**        | \-   | Area / Bounding box area                      |
| **Perimeter**     | μm   | Contour length                                |
| **Volume**        | μm³  | Estimated from cylindrical model              |
| **Median Radius** | μm   | Median distance from centroid to contour      |

### Installation

#### Prerequisites

We recommend using `Conda` for environment management to avoid dependency conflicts.

``` bash
# Create new environment
conda create -n bacterial-morph python=3.10 -y
conda activate bacterial-morph

# Install PyTorch (select appropriate CUDA version or CPU-only)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
# For CPU-only:
# conda install pytorch torchvision cpuonly -c pytorch -y

# Python dependencies
conda install opencv matplotlib scipy numpy pillow pygame -c conda-forge -y

# Segment Anything Model 
pip install git+https://github.com/facebookresearch/segment-anything.git
```

#### SAM Model Checkpoint

Download the SAM checkpoint and update the path in `morphology.py`:

``` bash
# Download ViT-H model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**Update path in `morphology.py`:**

``` python
sam_checkpoint = "/path/to/sam_vit_h_4b8939.pth"
```

### Usage

#### Box Detection: Automatic Detection (`box_auto.py`)

``` bash
python box_auto.py --input image.tif --output_dir output --prefix output_prefix \
--min_area 50 --max_area 2000 --block_size 21 --c_value 4 --dist_coeff 0.3
```

**Key Parameters:**

| Parameter           | Default | Description                                   |
|------------------------|--------------------|----------------------------|
| `--block_size`      | 21      | Adaptive thresholding block size (odd number) |
| `--c_value`         | 4       | Subtraction constant for thresholding         |
| `--dist_coeff`      | 0.3     | Distance transform threshold (0.0-1.0)        |
| `--min_area`        | 30      | Minimum cell area in pixels                   |
| `--max_area`        | 1000    | Maximum cell area in pixels                   |
| `--padding`         | 5       | Box expansion padding in pixels               |
| `--blank_threshold` | 200     | Brightness threshold to exclude empty regions |

**Advanced Filtering:**

``` bash
# Exclude top region 
python box_auto.py -i image.tif --exclude_y 0.03

# Exclude left region
python box_auto.py -i image.tif --exclude_x 0.01
```

#### Box Detection: Manual Selection (`box_manual.py`)

``` bash
python box_manual.py image.tif output/manual_boxes.txt
```

**Controls:**

| Action            | Operation                              |
|-------------------|----------------------------------------|
| Left Click + Drag | Draw bounding box around cell          |
| Right Click       | Add annotation point with custom label |
| Space             | Skip current cell (assigns index only) |
| Close Window      | Save and exit                          |

#### Morphological Analysis (`morphology.py`)

``` bash
python morphology.py image.tif box.txt output_prefix
```

**Arguments:** 1. Input image path 2. Box coordinates file (from Step 1) 3. Output prefix for result files

#### Example Output: `analysis_results_size.txt`

    cell_number area    width   length  aspect_ratio    eccentricity    circularity extent  perimeter   centroid
    1   12.4567 1.2345  3.4567  2.8012  0.8567  0.7234  0.8912  8.9234  0.5678
    2   15.6789 1.4567  3.8901  2.6705  0.8234  0.7456  0.8567  10.1234 0.6789
    3   8.9012  1.1234  2.5678  2.2857  0.7890  0.8123  0.9234  7.3456  0.4567

#### Scale Calibration

Default calibration in `morphology.py`:

``` python
um_per_px = 9 / 118  # 9 μm scale bar = 118 pixels
```

**To customize:** Measure your scale bar in pixels and update:

``` python
um_per_px = actual_length_um / measured_pixels
```

#### Box Coordinate Format

All box files use tab-separated values:

    index   x1  y1  x2  y2

-   `(x1, y1)`: Top-left corner
-   `(x2, y2)`: Bottom-right corner
-   Coordinates in pixel units (0,0 at top-left)

#### Acknowledgments

-   [Meta AI Segment Anything Model](https://github.com/facebookresearch/segment-anything) for state-of-the-art segmentation
-   OpenCV community for computer vision algorithms



------------------------------------------------------------------------




------------------------------------------------------------------------


## Models and Data

### Datasets

The `data` directory contains single-cell morphological and spectral data. These data were collected from 25 bacterial species spanning 14 genera commonly found in the human microbiome:

-   *Akkermansia muciniphila*
-   *Bacillus licheniformis*
-   *Bacteroides cellulosilyticus*
-   *Bacteroides fragilis*
-   *Bacteroides thetaiotaomicron*
-   *Bacteroides uniformis*
-   *Bifidobacterium adolescentis*
-   *Bifidobacterium breve*
-   *Bifidobacterium longum*
-   *Bifidobacterium pseudocatenulatum*
-   *Clostridium butyricum*
-   *Clostridium perfringens*
-   *Clostridium tertium*
-   *Enterococcus faecalis*
-   *Enterococcus faecium*
-   *Escherichia coli*
-   *Klebsiella pneumoniae*
-   *Lacticaseibacillus paracasei*
-   *Lactiplantibacillus plantarum*
-   *Lactococcus lactis*
-   *Staphylococcus capitis*
-   *Staphylococcus epidermidis*
-   *Streptococcus mitis*
-   *Streptococcus oralis*
-   *Veillonella atypica*

### Model Weights

The pre-trained model weights for the deep neural network and elastic net models are available [here](https://figshare.com/s/02981a4786792ae9052f). Pre-computed reference feature database is available [here](https://figshare.com/s/10a4129ad516fbc710c7).
