# Example

**Prerequisites**: Please ensure [scCulturePrec](https://github.com/MicrobeLab/scCulturePrec) is installed before use. `dnn_demo.pth` is provided on [figshare](https://figshare.com/s/02981a4786792ae9052f).

Step 1: Obtaining distances to reference samples

```{bash}
scCulturePrec dl-model --fn test_input.npy --out test_output \
--embed-size-spectra 128 --embed-size-morphol 128 --in_type both --input-dir example \
--num-morphol 10 --weight /path/to/dnn_demo.pth --feat-db /path/to/features_demo.h5

```

Step 2: Predicting whether the test samples belong to the same taxon as the reference samples

```{bash}
scCulturePrec elastic-net --model-file elastic_demo.pkl --dist-new test_output_dist.txt \
--pred-out predictions.csv
```
