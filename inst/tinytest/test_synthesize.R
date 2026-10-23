library(tabicl2synthr)

# --- Schema round-trip without Python --------------------------------------

d <- data.frame(
  x = c(1.5, 2.5, 3.5),
  i = 4:6L,
  b = c(TRUE, FALSE, NA),
  g = factor(c("a", "b", "a"), levels = c("a", "b", "c")),
  o = ordered(c("hi", "lo", "hi"), levels = c("lo", "hi")),
  stringsAsFactors = FALSE
)
enc <- tabicl2synthr:::.synthr_encode(d)
expect_true(is.matrix(enc$x) && storage.mode(enc$x) == "double")
expect_equal(dim(enc$x), c(3, 5))
expect_equal(enc$x[, "i"], c(4, 5, 6))
expect_equal(enc$x[, "b"], c(1, 0, NA))
expect_equal(enc$x[, "g"], c(0, 1, 0))          # 0-based codes
expect_equal(enc$x[, "o"], c(1, 0, 1))
expect_equal(unname(enc$categorical), c(FALSE, FALSE, TRUE, TRUE, TRUE))
back <- tabicl2synthr:::.synthr_decode(enc$x, enc)
expect_identical(names(back), names(d))
expect_equal(back$x, d$x)
expect_identical(back$i, d$i)
expect_identical(back$b, d$b)
expect_identical(levels(back$g), levels(d$g))
expect_identical(back$g, d$g)
expect_true(is.ordered(back$o))
expect_identical(back$o, d$o)

# Character columns become factors (with a message).
expect_message(enc_chr <- tabicl2synthr:::.synthr_encode(
  data.frame(s = c("p", "q", "p"))), "character")
expect_equal(enc_chr$x[, 1], c(0, 1, 0))
expect_true(enc_chr$categorical)

# Out-of-range codes are clipped, integers are rounded on decode.
odd <- tabicl2synthr:::.synthr_decode(
  cbind(c(-3, 9), c(1.6, -0.4)),
  list(names = c("g", "i"),
       encoding = list(g = list(type = "factor", levels = c("a", "b")),
                       i = list(type = "integer")),
       categorical = c(g = TRUE, i = FALSE)))
expect_identical(odd$g, factor(c("a", "b"), levels = c("a", "b")))
expect_identical(odd$i, c(2L, 0L))

# Input validation.
expect_error(tabicl2synthr:::.synthr_encode(1:3), "data.frame")
expect_error(tabicl2synthr:::.synthr_encode(data.frame(a = 1:2)[, 0]),
  "at least one column")
expect_error(tabicl2synthr:::.synthr_encode(iris[0, ]), "at least one row")
expect_error(tabicl2synthr:::.synthr_encode(data.frame(a = 1:2, b = Sys.Date())),
  "not supported")
expect_error(tabicl_synthesize(d, n_samples = 0), "n_samples")
expect_error(tabicl_synthesize(d, temperature = -1), "temperature")
expect_error(tabicl_synthesize(d, device = 1), "device")

# --- Mock Python module ------------------------------------------------------

# API-compatible stand-in for tabicl.TabICLUnsupervised: exercises the real
# encode -> constructor -> fit -> generate path without Python or checkpoints.
make_api <- function(calls = new.env(parent = emptyenv())) {
  calls$constructor <- list()
  calls$fit <- list()
  calls$generate <- list()
  unsupervised <- function(n_estimators, categorical_features, batch_size,
                           random_state, device, ...) {
    calls$constructor <- list(n_estimators = n_estimators,
      categorical_features = categorical_features, batch_size = batch_size,
      random_state = random_state, device = device)
    model <- new.env()
    model$fit <- function(x) {
      calls$fit <- list(x)
      model$x <- x
      invisible(NULL)
    }
    model$generate <- function(n_samples, temperature) {
      calls$generate <- list(n_samples = n_samples, temperature = temperature)
      out <- model$x[rep(seq_len(nrow(model$x)), length.out = n_samples), ,
        drop = FALSE]
      # Fractional offsets on categorical codes test the rounding in decode.
      if (!is.null(categorical_features))
        out[, categorical_features + 1L] <- out[, categorical_features + 1L] + 0.4
      out
    }
    model$impute <- function(x, temperature, n_iterations) {
      calls$impute <- list(nrow(x) == nrow(model$x), temperature = temperature,
        n_iterations = n_iterations)
      for (j in seq_len(ncol(x))) {
        na <- is.na(x[, j])
        if (any(na)) x[na, j] <- if (j %in% categorical_features) 0 else
          median(x[, j], na.rm = TRUE)
      }
      x
    }
    model$score_samples <- function(x, n_permutations) {
      calls$score <- list(nrow(x), n_permutations)
      seq_len(nrow(x)) / 10
    }
    model
  }
  list(module = list(TabICLUnsupervised = unsupervised), calls = calls)
}

api <- make_api()
d2 <- d[-3, ]    # drop the NA row: real generation never produces missings
res <- tabicl_synthesize(d2, n_samples = 7, temperature = 0.5, n_estimators = 2L,
  device = "cpu", random_state = 7L, tabicl_module = api$module)
ctor <- api$calls$constructor
expect_equal(ctor$n_estimators, 2L)
expect_equal(ctor$categorical_features, c(2L, 3L, 4L))  # 0-based: b, g, o
expect_equal(ctor$random_state, 7L)
expect_equal(ctor$device, "cpu")
expect_null(ctor$batch_size)
expect_true(is.matrix(api$calls$fit[[1]]))
expect_equal(api$calls$generate, list(n_samples = 7L, temperature = 0.5))
expect_equal(nrow(res), 7L)
expect_identical(names(res), names(d))
expect_true(is.numeric(res$x) && is.integer(res$i) && is.logical(res$b))
expect_identical(levels(res$g), levels(d$g))
expect_true(is.ordered(res$o))
expect_false(anyNA(res))
expect_true(all(res$g %in% levels(d$g)))

# n_samples defaults to nrow(data).
api2 <- make_api()
res_default <- tabicl_synthesize(d, tabicl_module = api2$module)
expect_equal(nrow(res_default), nrow(d))
expect_equal(api2$calls$constructor$categorical_features, c(2L, 3L, 4L))

# Imputation: mock fills missings; observed cells must survive untouched.
dm <- d
dm$x[1] <- NA
dm$g[2] <- NA
api3 <- make_api()
imp <- tabicl_impute(dm, temperature = 0.2, n_iterations = 3L,
  tabicl_module = api3$module)
expect_false(anyNA(imp))
expect_equal(imp$x[-1], dm$x[-1])
expect_identical(imp$g[1], dm$g[1])
expect_identical(levels(imp$g), levels(dm$g))
expect_equal(api3$calls$impute$temperature, 0.2)
expect_equal(api3$calls$impute$n_iterations, 3L)

# Without missings the Python impute step is bypassed entirely.
api4 <- make_api()
expect_equal(tabicl_impute(d2, tabicl_module = api4$module), d2)

# Scoring.
api5 <- make_api()
scores <- tabicl_score_samples(d, n_permutations = 3L, tabicl_module = api5$module)
expect_equal(scores, seq_len(nrow(d)) / 10)
expect_equal(api5$calls$score[[2]], 3L)
expect_error(tabicl_score_samples(d, newdata = d[, -1],
  tabicl_module = api5$module), "same columns")
