## ----setup, include = FALSE---------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----cran-install, eval = FALSE-----------------------------------
# install.packages("fastPLS")

## ----github-install, eval = FALSE---------------------------------
# install.packages("remotes")
# remotes::install_github(
#     "tkcaccia/fastPLS",
#     upgrade = "never",
#     force = TRUE,
#     build_vignettes = TRUE
# )

## ----macos-install, eval = FALSE----------------------------------
# Sys.setenv(FASTPLS_USE_METAL = "1")
# remotes::install_github(
#     "tkcaccia/fastPLS",
#     force = TRUE,
#     upgrade = "never",
#     build_vignettes = TRUE
# )

## ----linux-openblas, eval = FALSE---------------------------------
# Sys.setenv(FASTPLS_USE_OPENBLAS = "1")
# install.packages("fastPLS", type = "source")

## ----windows-openblas-install, eval = FALSE-----------------------
# Sys.setenv(
#     FASTPLS_USE_OPENBLAS = "1",
#     OPENBLAS_ROOT = "C:/msys64/ucrt64"
# )
# remotes::install_github(
#     "tkcaccia/fastPLS",
#     force = TRUE,
#     upgrade = "never",
#     build_vignettes = TRUE
# )

## ----windows-arm64, eval = FALSE----------------------------------
# Sys.unsetenv(c("OPENBLAS_ROOT", "FASTPLS_USE_OPENBLAS"))
# remotes::install_github(
#     "tkcaccia/fastPLS",
#     force = TRUE,
#     upgrade = "never",
#     build_vignettes = TRUE
# )

## ----cuda-install, eval = FALSE-----------------------------------
# Sys.setenv(
#     FASTPLS_USE_CUDA = "1",
#     FASTPLS_REQUIRE_CUDA = "1",
#     CUDA_HOME = "/usr/local/cuda"
# )
# install.packages("fastPLS", type = "source")

## ----cuda-windows, eval = FALSE-----------------------------------
# Sys.setenv(
#     CUDA_ROOT = paste0(
#         "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
#         "v12.6"
#     )
# )

## ----verify-installation, eval = FALSE----------------------------
# library(fastPLS)
# 
# fastPLS_blas()
# cuda_info()
# has_cuda()
# has_metal()
# packageVersion("fastPLS")

## ----verify-fit, eval = FALSE-------------------------------------
# X <- as.matrix(iris[, 1:4])
# y <- iris$Species
# 
# fit <- pls(X, y, ncomp = 1:2, backend = "cpu", seed = 11)
# stopifnot(length(fit$Yfit) == 2L)
# 
# if (has_cuda()) {
#     fit_cuda <- pls(X, y, ncomp = 1:2, backend = "cuda", seed = 11)
# }
# 
# if (has_metal()) {
#     fit_metal <- pls(X, y, ncomp = 1:2, backend = "metal", seed = 11)
# }

## ----backend-options, eval = FALSE--------------------------------
# options(backend = "cpu", n.cores = 4L)
# 
# fit <- pls(X, y, ncomp = 1:2)
# fit_one_core <- pls(X, y, ncomp = 1:2, n.cores = 1L)

## ----clear-windows-architecture-settings, eval = FALSE------------
# Sys.unsetenv(c(
#     "OPENBLAS_ROOT",
#     "R_TOOLS_SOFT",
#     "CUDA_ROOT",
#     "CUDA_PATH"
# ))

## ----open-vignettes, eval = FALSE---------------------------------
# vignette("installation", package = "fastPLS")
# vignette("fastPLS", package = "fastPLS")

## ----reproducibility, eval = FALSE--------------------------------
# fastPLS_blas()
# cuda_info()
# has_cuda()
# has_metal()

## ----session-information------------------------------------------
sessionInfo()

