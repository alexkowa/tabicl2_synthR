# Opt-in end-to-end tests against the real Python stack. Run in a configured
# environment (tabicl 2.2.x installed, checkpoints cached); routine package
# checks never initialize Python or download anything.
python <- Sys.getenv("TABICLR_TEST_PYTHON", "")
if (!nzchar(python)) exit_file("set TABICLR_TEST_PYTHON to a Python with tabicl 2.2")
if (!requireNamespace("reticulate", quietly = TRUE)) exit_file("no reticulate")
# reticulate ignores use_python() once RETICULATE_PYTHON is set, so override it.
Sys.setenv(RETICULATE_PYTHON = python)
version <- tryCatch(reticulate::import("importlib.metadata")$version("tabicl"),
  error = function(e) NULL)
if (is.null(version)) exit_file("python tabicl not installed")

library(tabicl2synthr)
reticulate::import("torch")$set_num_threads(1L)
set.seed(1)
train <- head(iris, 60)
synth <- tabicl_synthesize(train, n_samples = 25, n_estimators = 2L,
  device = "cpu", random_state = 1L, python = python)
expect_identical(names(synth), names(train))
expect_equal(nrow(synth), 25L)
expect_true(is.numeric(synth$Sepal.Length))
expect_identical(levels(synth$Species), levels(train$Species))
expect_false(anyNA(synth))
# Generated data should be new rows, not copies of the training rows.
expect_false(any(duplicated(rbind(train[1:4], synth[1:25, 1:4]))))

mixed <- data.frame(
  a = rnorm(40), b = sample.int(3, 40, TRUE), c = sample(c(TRUE, FALSE), 40, TRUE),
  g = factor(sample(letters[1:3], 40, TRUE)))
mixed_synth <- tabicl_synthesize(mixed, n_samples = 10, n_estimators = 2L,
  device = "cpu", python = python)
expect_true(is.integer(mixed_synth$b) && is.logical(mixed_synth$c))
expect_identical(levels(mixed_synth$g), levels(mixed$g))

miss <- mixed
miss$a[3] <- NA_real_
miss$g[7] <- NA
imp <- tabicl_impute(miss, n_estimators = 2L, device = "cpu", python = python)
expect_false(anyNA(imp))
expect_equal(imp$a[-3], miss$a[-3])
expect_identical(imp$b, miss$b)

scores <- tabicl_score_samples(mixed, newdata = miss, n_estimators = 2L,
  n_permutations = 2L, device = "cpu", python = python)
expect_equal(length(scores), nrow(miss))
expect_true(all(is.finite(scores)))
