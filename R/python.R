# Lazy Python access: nothing below touches reticulate or Python until a
# user actually calls a wrapper function.

# Python expects categorical_features as an iterable; a length-1 R vector is
# converted to a scalar int, so always go through r_to_py(). NULL keeps
# upstream auto-detection. R mocks keep receiving plain integer vectors.
.synthr_py_index_list <- function(indices, python = TRUE) {
  if (!length(indices)) return(NULL)
  if (!python) return(as.integer(indices))
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("tabicl2synthr requires the optional R package 'reticulate'.")
  reticulate::r_to_py(as.list(as.integer(indices)))
}

# Check the Python tabicl installation and return the module (or an injected
# API-compatible replacement for testing). Mirrors VIM's tabicl import guard.
.tabicl_import <- function(python = NULL, tabicl_module = NULL) {
  if (!is.null(tabicl_module)) return(tabicl_module)
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("tabicl2synthr requires the optional R package 'reticulate'.")
  if (!is.null(python)) reticulate::use_python(python, required = TRUE)
  tryCatch({
    version <- reticulate::import("importlib.metadata")$version("tabicl")
    if (utils::compareVersion(version, "2.2.0") < 0L ||
        utils::compareVersion(version, "2.3.0") >= 0L)
      stop("Expected TabICL 2.2.x; found ", version)
    reticulate::import("tabicl", convert = TRUE)
  }, error = function(e) stop("TabICL 2.2 requires Python >= 3.10 and ",
    "'tabicl==2.2.0'. Install inst/python/tabicl-requirements.txt in ",
    "reticulate's Python environment.\n", conditionMessage(e), call. = FALSE))
}

# Encode a data frame as the numeric matrix TabICLUnsupervised expects.
# Factors/ordered/logical become 0-based integer codes (TabICL stores
# categorical values as non-negative integers); integers become doubles.
# Returns everything .synthr_decode() needs to restore the original schema.
.synthr_encode <- function(data) {
  if (!is.data.frame(data)) stop("data must be a data.frame")
  data <- as.data.frame(data, stringsAsFactors = FALSE)
  if (!ncol(data)) stop("data must have at least one column")
  if (!nrow(data)) stop("data must have at least one row")
  encoding <- list()
  cols <- lapply(names(data), function(nm) {
    x <- data[[nm]]
    if (is.character(x) && !is.factor(x)) {
      message("Converting character column '", nm, "' to a factor.")
      x <- factor(x)
    }
    if (is.factor(x)) {
      encoding[[nm]] <<- list(type = if (is.ordered(x)) "ordered" else "factor",
                              levels = levels(x))
      as.numeric(as.integer(x)) - 1
    } else if (is.logical(x)) {
      encoding[[nm]] <<- list(type = "logical")
      as.numeric(x)
    } else if (is.integer(x)) {
      encoding[[nm]] <<- list(type = "integer")
      as.numeric(x)
    } else if (is.numeric(x)) {
      encoding[[nm]] <<- list(type = "double")
      as.numeric(x)
    } else {
      stop("Column '", nm, "' of type '", class(x)[1], "' is not supported")
    }
  })
  names(cols) <- names(data)
  x <- do.call(cbind, cols)
  storage.mode(x) <- "double"
  rownames(x) <- NULL
  categorical <- vapply(names(data),
    function(nm) encoding[[nm]]$type %in% c("factor", "ordered", "logical"),
    logical(1))
  list(x = x, names = names(data), encoding = encoding, categorical = categorical)
}

# Turn the numeric matrix returned by TabICLUnsupervised back into a data
# frame with the types and levels recorded by .synthr_encode().
.synthr_decode <- function(synth, encoded) {
  synth <- as.matrix(synth)
  storage.mode(synth) <- "double"
  if (ncol(synth) != length(encoded$names))
    stop("TabICL returned ", ncol(synth), " columns, expected ",
      length(encoded$names))
  out <- lapply(seq_along(encoded$names), function(j) {
    col <- synth[, j]
    info <- encoded$encoding[[encoded$names[j]]]
    switch(info$type,
      double = as.numeric(col),
      integer = as.integer(round(col)),
      logical = as.logical(pmin(pmax(round(col), 0), 1)),
      {
        # Factor and ordered: sample class codes, clip to the schema so that
        # stray values outside the training levels become valid levels.
        k <- length(info$levels)
        code <- pmin(pmax(round(col), 0), k - 1L)
        vals <- info$levels[code + 1L]
        if (identical(info$type, "ordered"))
          ordered(vals, levels = info$levels)
        else factor(vals, levels = info$levels)
      })
  })
  names(out) <- encoded$names
  structure(out, class = "data.frame", row.names = seq_along(out[[1]]))
}

# Validate a scalar option; returns it cast to `cast`. Options must be single
# non-missing numbers of at least `min`; whole numbers if `whole = TRUE`.
.synthr_scalar <- function(value, name, cast, min = -Inf, whole = FALSE) {
  if (is.null(value)) return(NULL)
  bad <- !(is.numeric(value) || is.integer(value)) || length(value) != 1L ||
    is.na(value) || value < min || (whole && value != round(value))
  if (bad) {
    msg <- paste0("'", name, "' must be a single number >= ", min)
    stop(if (whole) paste(msg, "and a whole number") else msg)
  }
  cast(value)
}
