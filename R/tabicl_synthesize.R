#' Generate synthetic tabular data with TabICLv2
#'
#' Generate a synthetic data set that mimics an input data set. The columns are
#' modelled jointly through TabICLUnsupervised from the
#' Python `tabicl` package: the pretrained TabICLv2 model estimates each
#' column conditional on the preceding ones, and new rows are drawn by
#' autoregressive sampling (Qu et al., 2026).
#'
#' Python is initialized lazily on the first call. Install `reticulate` and
#' `python -m pip install -r inst/python/tabicl-requirements.txt` (i.e.
#' `tabicl==2.2.0`, Python >= 3.10) in its Python environment. The TabICLv2
#' model checkpoints are downloaded from Hugging Face on first use.
#'
#' Numeric, integer, logical and factor columns are supported; character
#' columns are converted to factors. Missing values in `data` are passed to
#' the model, which handles them natively, but the generated data set
#' contains no missings. Factors become Python categoricals (integer codes
#' with the schema kept fixed at the training levels), so synthetic rows only
#' contain observed levels. Runtime grows with the number of columns and
#' rows, because every column requires one conditional TabICL pass; a GPU
#' substantially speeds this up (see `device`).
#'
#' @param data A data frame with at least one row and column. Numeric,
#'   integer, logical, factor, ordered and character columns are supported.
#' @param n_samples Number of synthetic rows to generate. Defaults to
#'   `nrow(data)`.
#' @param temperature Non-negative scaling of the predictive distributions.
#'   `0` yields the conditional medians/modes, `1` (the default) samples from
#'   the full distributions, and values `> 1` increase diversity at the cost
#'   of realism.
#' @param n_estimators Number of TabICL ensemble members per conditional
#'   prediction. More members improve accuracy but are slower.
#' @param device Computation device, e.g. `"cpu"` or `"cuda"`. `NULL`
#'   (the default) lets TabICL auto-select CUDA, XPU, MPS or CPU.
#' @param random_state Integer seed for reproducibility. Note that the Python
#'   side is not fully deterministic on GPU.
#' @param python Optional path to the Python interpreter to use
#'   (passed to [reticulate::use_python()]). Note that reticulate ignores
#'   this choice when the `RETICULATE_PYTHON` environment variable is set.
#' @param tabicl_module Optional API-compatible replacement for the Python
#'   `tabicl` module, for testing only.
#' @return A data frame with `n_samples` rows and the same columns, types and
#'   factor levels as `data`.
#' @export
#' @seealso [tabicl_impute()] and [tabicl_score_samples()] for the imputation
#'   and outlier-scoring counterparts.
#' @references
#' Qu, J., Holzmüller, D., Le Morvan, M., Varoquaux, G. (2026).
#' TabICLv2: A better, faster, scalable, and open tabular foundation model.
#' arXiv:2602.11139.
#' @examples
#' \dontrun{
#' synth <- tabicl_synthesize(iris, n_samples = 50, device = "cpu")
#' str(synth)
#' }
tabicl_synthesize <- function(data, n_samples = nrow(data), temperature = 1,
                              n_estimators = 8, device = NULL,
                              random_state = 42L, python = NULL,
                              tabicl_module = NULL) {
  encoded <- .synthr_encode(data)
  n_samples <- .synthr_scalar(n_samples, "n_samples", as.integer, min = 1, whole = TRUE)
  temperature <- .synthr_scalar(temperature, "temperature", as.double, min = 0)
  n_estimators <- .synthr_scalar(n_estimators, "n_estimators", as.integer,
    min = 1, whole = TRUE)
  random_state <- .synthr_scalar(random_state, "random_state", as.integer,
    min = 0, whole = TRUE)
  if (!is.null(device) && !(is.character(device) && length(device) == 1L))
    stop("'device' must be a single string or NULL")
  tabicl <- .tabicl_import(python, tabicl_module)
  if (is.null(tabicl$TabICLUnsupervised))
    stop("The Python tabicl module does not expose TabICLUnsupervised")
  model <- tabicl$TabICLUnsupervised(
    n_estimators = n_estimators,
    categorical_features = .synthr_py_index_list(
      which(encoded$categorical) - 1L, python = is.null(tabicl_module)),
    batch_size = NULL, random_state = random_state, device = device)
  model$fit(encoded$x)
  out <- model$generate(n_samples = n_samples, temperature = temperature)
  .synthr_decode(out, encoded)
}

# Internal: fit TabICLUnsupervised on data and return list(model, encoding)
# shared by tabicl_impute() and tabicl_score_samples().
.tabicl_unsupervised_fit <- function(data, n_estimators, device, random_state,
                                     python, tabicl_module = NULL) {
  encoded <- .synthr_encode(data)
  n_estimators <- .synthr_scalar(n_estimators, "n_estimators", as.integer,
    min = 1, whole = TRUE)
  random_state <- .synthr_scalar(random_state, "random_state", as.integer,
    min = 0, whole = TRUE)
  if (!is.null(device) && !(is.character(device) && length(device) == 1L))
    stop("'device' must be a single string or NULL")
  tabicl <- .tabicl_import(python, tabicl_module)
  if (is.null(tabicl$TabICLUnsupervised))
    stop("The Python tabicl module does not expose TabICLUnsupervised")
  model <- tabicl$TabICLUnsupervised(
    n_estimators = n_estimators,
    categorical_features = .synthr_py_index_list(
      which(encoded$categorical) - 1L, python = is.null(tabicl_module)),
    batch_size = NULL, random_state = random_state, device = device)
  model$fit(encoded$x)
  list(model = model, encoded = encoded)
}

#' Impute missing values with TabICLv2
#'
#' Fill missing values in a data set by conditioning on the observed columns
#' with TabICLUnsupervised from the Python `tabicl`
#' package, mirroring [tabicl_synthesize()]. See its documentation for the
#' Python setup, supported column types and runtime behaviour.
#'
#' @inheritParams tabicl_synthesize
#' @param data A data frame, possibly with missing values.
#' @param temperature Non-negative scaling of the predictive distributions.
#'   Near `0` (the default) gives deterministic median/mode imputations;
#'   `1` samples from the full distributions for imputation uncertainty.
#' @param n_iterations Number of iterative refinement passes over the
#'   incomplete columns.
#' @return A data frame with the same dimensions and schema as `data` and all
#'   missing values filled.
#' @export
#' @examples
#' \dontrun{
#' d <- iris
#' d$Sepal.Length[c(1, 5, 12)] <- NA
#' tabicl_impute(d, device = "cpu")
#' }
tabicl_impute <- function(data, temperature = 0, n_iterations = 2,
                          n_estimators = 8, device = NULL,
                          random_state = 42L, python = NULL,
                          tabicl_module = NULL) {
  temperature <- .synthr_scalar(temperature, "temperature", as.double, min = 0)
  n_iterations <- .synthr_scalar(n_iterations, "n_iterations", as.integer,
    min = 1, whole = TRUE)
  fit <- .tabicl_unsupervised_fit(data, n_estimators, device, random_state,
    python, tabicl_module)
  missings <- is.na(fit$encoded$x)
  if (!any(missings)) return(.synthr_decode(fit$encoded$x, fit$encoded))
  filled <- fit$model$impute(fit$encoded$x, temperature = temperature,
    n_iterations = n_iterations)
  result <- .synthr_decode(filled, fit$encoded)
  # Observed cells keep their original values (the encoded matrix is an exact
  # numeric copy), so imputation never alters observed data.
  original <- .synthr_decode(fit$encoded$x, fit$encoded)
  for (j in seq_along(result)) {
    column <- result[[j]]
    column[!missings[, j]] <- original[[j]][!missings[, j]]
    result[[j]] <- column
  }
  result
}

#' Score samples by outlierness with TabICLv2
#'
#' Estimate how typical each row of a data set is under the joint distribution
#' learned from a reference data set with TabICLUnsupervised
#' from the Python `tabicl` package. Scores are exponentiated, permutation-
#' averaged log-densities: higher values indicate more typical rows, lower
#' values indicate outliers. See [tabicl_synthesize()] for the Python setup
#' and supported column types.
#'
#' @inheritParams tabicl_synthesize
#' @param data Data frame used as the conditioning (training) data set.
#' @param newdata Data frame with the same columns to score. Defaults to
#'   `data` (in-sample scores).
#' @param n_permutations Number of random feature orderings to average over.
#' @return A numeric vector of length `nrow(newdata)` with the outlier scores.
#' @export
#' @examples
#' \dontrun{
#' scores <- tabicl_score_samples(iris, device = "cpu")
#' summary(scores)
#' }
tabicl_score_samples <- function(data, newdata = data, n_permutations = 4,
                                 n_estimators = 8, device = NULL,
                                 random_state = 42L, python = NULL,
                                 tabicl_module = NULL) {
  n_permutations <- .synthr_scalar(n_permutations, "n_permutations", as.integer,
    min = 1, whole = TRUE)
  fit <- .tabicl_unsupervised_fit(data, n_estimators, device, random_state,
    python, tabicl_module)
  scored <- .synthr_encode(newdata)
  if (!identical(scored$names, fit$encoded$names))
    stop("newdata must have the same columns in the same order as data")
  as.numeric(fit$model$score_samples(scored$x, n_permutations = n_permutations))
}
