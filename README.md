# tabicl2synthr

Synthetic tabular data generation in R based on an input data set, using the
[TabICLv2](https://github.com/soda-inria/tabicl) tabular foundation model.

`tabicl_synthesize()` fits the unsupervised mode of the Python `tabicl`
package (`TabICLUnsupervised`) on your data and draws new rows by
autoregressive sampling of each column conditional on the preceding ones.
Two related wrappers expose the rest of the same model: `tabicl_impute()`
(condition-based imputation) and `tabicl_score_samples()` (outlier scores).

## Installation

```r
# From R, once the repository is cloned:
install.packages("devtools")
devtools::install("tabicl2_synthR")
```

## Python setup

The package calls Python lazily through
[reticulate](https://rstudio.github.io/reticulate/). Install `tabicl` (>= 2.2,
< 2.3) in the Python environment reticulate uses:

```sh
python -m pip install -r inst/python/tabicl-requirements.txt
```

This pulls in `torch`, `scikit-learn` and friends (Python >= 3.10). The
pretrained TabICLv2 checkpoints are downloaded from Hugging Face on first use.

## Usage

```r
library(tabicl2synthr)

synth <- tabicl_synthesize(iris, n_samples = 200, device = "cpu")
str(synth)   # same columns, types and factor levels as iris

d <- airquality
synth_air <- tabicl_synthesize(d)   # NA handling is native to the model
```

## Notes

- Supported column types: numeric, integer, logical, factor, ordered,
  character (converted to factor).
- `temperature = 0` gives conditional medians/modes; `1` samples from the
  full predictive distributions (default); larger values increase diversity.
- Every column is one conditional TabICL pass, so runtime grows with the
  number of columns and training rows. A GPU (`device = "cuda"`) helps a lot.

## Testing

`R CMD check` runs the mock-based test suite (no Python needed). The
end-to-end tests against the real Python model are opt-in:

```sh
TABICLR_TEST_PYTHON=/path/to/python-with-tabicl \
  Rscript -e 'tinytest::run_test_dir("inst/tinytest")'
```

## License

GPL-3. The wrapped Python `tabicl` package has its own (permissive) license.
