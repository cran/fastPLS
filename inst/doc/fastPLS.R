## ----setup, include = FALSE---------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
options(width = 68)
library(fastPLS)

## ----install-cran, eval = FALSE-----------------------------------
# install.packages("fastPLS")

## ----install-github, eval = FALSE---------------------------------
# install.packages("remotes")
# remotes::install_github(
#     "tkcaccia/fastPLS",
#     upgrade = "never",
#     force = TRUE,
#     build_vignettes = TRUE
# )

## ----require-metal, eval = FALSE----------------------------------
# Sys.setenv(FASTPLS_USE_METAL = "1")

## ----require-openblas-linux, eval = FALSE-------------------------
# Sys.setenv(FASTPLS_USE_OPENBLAS = "1")
# install.packages("fastPLS", type = "source")

## ----require-openblas-windows, eval = FALSE-----------------------
# Sys.setenv(
#     FASTPLS_USE_OPENBLAS = "1",
#     OPENBLAS_ROOT = "C:/msys64/ucrt64"
# )
# remotes::install_github(
#     "tkcaccia/fastPLS",
#     upgrade = "never",
#     force = TRUE,
#     build_vignettes = TRUE
# )

## ----verify-compiled-libraries, eval = FALSE----------------------
# library(fastPLS)
# 
# fastPLS_blas()
# has_cuda()
# has_metal()

## ----chunk-002----------------------------------------------------
set.seed(100)
X <- as.matrix(iris[, 1:4])
Y_cls <- iris$Species
cls_test_id <- sample(seq_len(nrow(X)), 30)
Xtrain <- X[-cls_test_id, , drop = FALSE]
Xtest <- X[cls_test_id, , drop = FALSE]
Ytrain_cls <- Y_cls[-cls_test_id]
Ytest_cls <- Y_cls[cls_test_id]

## ----chunk-003----------------------------------------------------
fit_cls <- pls(
    Xtrain,
    Ytrain_cls,
    Xtest,
    Ytest_cls,
    ncomp = 1:2,
    fit = TRUE,
    return_variance = FALSE,
    seed = 101
)

fit_cls$accuracy

## ----chunk-004----------------------------------------------------
fit_cls_train_only <- pls(
    Xtrain,
    Ytrain_cls,
    ncomp = 1:2,
    classifier = "lda",
    fit = TRUE,
    return_variance = FALSE,
    seed = 101
)

pred_cls_later <- predict(
    fit_cls_train_only,
    Xtest,
    Ytest = Ytest_cls,
    top = 2,
    raw_scores = TRUE
)

pred_cls_later$accuracy
pred_cls_later$metrics$metrics
head(pred_cls_later$Ypred_top[["ncomp=2"]])

## ----chunk-005----------------------------------------------------
eval_cls_path <- evaluate(
    observed = Ytest_cls,
    predicted = pred_cls_later
)

eval_cls <- eval_cls_path$by_component[["ncomp=2"]]
eval_cls

## ----chunk-005a---------------------------------------------------
fit_cls$metrics$test[["ncomp=2"]]$metrics

## ----chunk-006----------------------------------------------------
eval_cls$confusion

## ----chunk-007----------------------------------------------------
score_last <- pred_cls_later$LDA_scores[
    , , dim(pred_cls_later$LDA_scores)[3L]
]

evaluate(
    observed = Ytest_cls,
    predicted = score_last
)

## ----chunk-008----------------------------------------------------
fit_cls_plssvd <- pls(
    Xtrain,
    Ytrain_cls,
    Xtest,
    Ytest_cls,
    ncomp = 1:2,
    method = "plssvd",
    seed = 100
)

head(fit_cls_plssvd$Ypred)

evaluate(
    observed = Ytest_cls,
    predicted = fit_cls_plssvd$Ypred[["ncomp=2"]]
)$confusion

## ----chunk-009, eval = FALSE--------------------------------------
# fit_cls_lda_gpu <- pls(
#     Xtrain,
#     Ytrain_cls,
#     Xtest,
#     Ytest_cls,
#     ncomp = 1:2,
#     method = "plssvd",
#     backend = "cuda",
#     classifier = "lda"
# )

## ----chunk-010----------------------------------------------------
fit_cls_lda_cpu <- pls(
    Xtrain,
    Ytrain_cls,
    Xtest,
    Ytest_cls,
    ncomp = 1:2,
    method = "plssvd",
    backend = "cpu",
    seed = 100,
    classifier = "lda"
)

head(fit_cls_lda_cpu$Ypred)

## ----chunk-012----------------------------------------------------
kernel_fits <- lapply(c("linear", "rbf", "poly"), function(k) {
    pls(
    Xtrain, Ytrain_cls, Xtest, Ytest_cls,
    ncomp = 1:2,
    method = "kernelpls",
    kernel = k,
    degree = 2,
    seed = 102
    )
})
names(kernel_fits) <- c("linear", "rbf", "poly")

kernel_accuracy <- vapply(kernel_fits, function(fit) {
    mean(fit$Ypred[["ncomp=2"]] == Ytest_cls)
}, numeric(1))
kernel_accuracy

## ----chunk-013, fig.width = 5, fig.height = 4---------------------
plot(
    fit_cls,
    groups = Ytrain_cls,
    ellipse = TRUE,
    ellipse.type = "confidence"
)

## ----iris-three-class-opls, fig.width = 7.5, fig.height = 4-------
opls_three <- pls(
    Xtrain,
    Ytrain_cls,
    Xtest,
    Ytest_cls,
    ncomp = 1:2,
    method = "opls",
    fit = TRUE,
    proj = TRUE,
    north = 2L,
    seed = 201
)

old_par <- par(mfrow = c(1, 2), mar = c(4.2, 4.2, 2.2, 0.8))
plot(
    opls_three,
    score.set = "train",
    groups = Ytrain_cls,
    ellipse = TRUE,
    ellipse.type = "hotelling",
    main = "OPLS training, north = 2"
)
plot(
    opls_three,
    score.set = "test",
    groups = opls_three$Ypred[["ncomp=2"]],
    xlim = c(-0.16, 0.42),
    main = "OPLS test prediction"
)
par(old_par)

## ----chunk-016----------------------------------------------------
set.seed(100)
Xreg <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec", "drat")])
Y_reg <- mtcars$mpg
reg_test_id <- sample(seq_len(nrow(Xreg)), 8)
Xreg_train <- Xreg[-reg_test_id, , drop = FALSE]
Xreg_test <- Xreg[reg_test_id, , drop = FALSE]
Ytrain_reg <- Y_reg[-reg_test_id]
Ytest_reg <- Y_reg[reg_test_id]

## ----chunk-017----------------------------------------------------
fit_reg <- pls(
    Xreg_train,
    Ytrain_reg,
    Xreg_test,
    Ytest_reg,
    ncomp = 1:3,
    fit = TRUE,
    return_variance = FALSE
)

fit_reg$Q2Y

## ----chunk-018----------------------------------------------------
Xreg32 <- float::fl(as.matrix(Xreg_train))
Yreg32 <- float::fl(matrix(Ytrain_reg, ncol = 1))
fit_reg32 <- pls(
    Xreg32,
    Yreg32,
    float::fl(as.matrix(Xreg_test)),
    float::fl(matrix(Ytest_reg, ncol = 1)),
    ncomp = 1:2
)
fit_reg32$Q2Y

## ----chunk-019, fig.width=4.8, fig.height=4-----------------------
reg_component <- dim(fit_reg$Ypred)[3L]
pred_mpg <- fit_reg$Ypred[, , reg_component]
plot(
    Ytest_reg,
    pred_mpg,
    pch = 21,
    bg = "#4E79A7",
    col = "black",
    xlab = "Observed mpg",
    ylab = "Predicted mpg",
    main = "Regression: observed vs predicted"
)
abline(0, 1, col = "#D55E00", lwd = 2)

## ----chunk-020----------------------------------------------------
fit_reg_train_only <- pls(
    Xreg_train,
    Ytrain_reg,
    ncomp = 1:3,
    fit = TRUE,
    return_variance = FALSE
)

pred_reg_later <- predict(
    fit_reg_train_only,
    Xreg_test,
    Ytest = Ytest_reg,
    proj = TRUE
)

pred_reg_later$Q2Y
pred_reg_later$metrics$metrics
head(pred_reg_later$Ttest)

## ----chunk-021----------------------------------------------------
eval_reg <- evaluate(
    observed = Ytest_reg,
    predicted = pred_mpg,
    ytrain = Ytrain_reg
)

eval_reg$task
eval_reg$metrics
lapply(eval_reg$metric_definitions, strwrap, width = 56)
eval_reg$per_response

## ----multivariate-regression--------------------------------------
Xmulti <- as.matrix(mtcars[, c("disp", "hp", "wt", "gear", "carb")])
Ymulti <- as.matrix(mtcars[, c("mpg", "qsec", "drat")])
Xmulti_train <- Xmulti[-reg_test_id, , drop = FALSE]
Xmulti_test <- Xmulti[reg_test_id, , drop = FALSE]
Ymulti_train <- Ymulti[-reg_test_id, , drop = FALSE]
Ymulti_test <- Ymulti[reg_test_id, , drop = FALSE]

fit_multi <- pls(
    Xtrain = Xmulti_train,
    Ytrain = Ymulti_train,
    Xtest = Xmulti_test,
    Ytest = Ymulti_test,
    ncomp = 1:3,
    method = "plssvd",
    bycol = TRUE,
    return_variance = FALSE,
    seed = 102
)

fit_multi$metrics$test[["ncomp=3"]]$metrics
fit_multi$metrics$test[["ncomp=3"]]$per_response

## ----multivariate-regression-cv-----------------------------------
cv_multi <- pls.single.cv(
    Xdata = Xmulti_train,
    Ydata = Ymulti_train,
    ncomp = 1:3,
    kfold = 3,
    method = "plssvd",
    fit = FALSE,
    bycol = TRUE,
    return_splits = TRUE,
    seed = 102
)

cv_multi$best_ncomp
cv_multi$best_metric_value
head(cv_multi$split_index)

## ----chunk-022----------------------------------------------------
fit_opls <- pls(
    Xreg_train,
    Ytrain_reg,
    Xreg_test,
    Ytest_reg,
    ncomp = 1:2,
    method = "opls",
    seed = 101
)

class(fit_opls)
fit_opls$Q2Y

## ----chunk-023----------------------------------------------------
cv_kfold <- pls.single.cv(
    Xdata = Xtrain,
    Ydata = Ytrain_cls,
    ncomp = 2,
    kfold = 5,
    return_splits = TRUE,
    seed = 103
)

cv_kfold$metrics$cross_validated[["ncomp=2"]]$metrics
head(cv_kfold$split_index)

## ----chunk-024----------------------------------------------------
patient_id <- rep(
    seq_len(ceiling(nrow(Xtrain) / 2)), each = 2
)[seq_len(nrow(Xtrain))]

cv_grouped <- pls.single.cv(
    Xdata = Xtrain,
    Ydata = Ytrain_cls,
    constrain = patient_id,
    ncomp = 2,
    kfold = 4,
    return_splits = TRUE,
    seed = 104
)

c(n_folds = length(unique(cv_grouped$fold)),
    n_patient_groups = length(unique(patient_id)))

head(data.frame(
    sample_index = seq_len(nrow(Xtrain)),
    patient_id = patient_id,
    cv_grouped$split_index,
    check.names = FALSE
), 8)

patient_rows <- split(seq_along(patient_id), patient_id)
patients_kept_together <- vapply(
    seq_len(ncol(cv_grouped$split_index)),
    function(fold) {
        all(vapply(patient_rows, function(rows) {
            length(unique(cv_grouped$split_index[rows, fold])) == 1L
        }, logical(1L)))
    },
    logical(1L)
)

stopifnot(all(patients_kept_together))
data.frame(
    fold = colnames(cv_grouped$split_index),
    patients_kept_together = patients_kept_together
)

## ----chunk-025----------------------------------------------------
cv_loocv <- pls.single.cv(
    Xdata = Xtrain,
    Ydata = Ytrain_cls,
    constrain = patient_id,
    ncomp = 1,
    kfold = "loocv",
    seed = 105
)

c(n_loocv_folds = length(unique(cv_loocv$fold)),
    n_patient_groups = length(unique(patient_id)))

## ----chunk-026----------------------------------------------------
cv_opt <- pls.single.cv(
    Xdata = Xreg_train,
    Ydata = Ytrain_reg,
    ncomp = 1:3,
    kfold = 5
)

cv_opt$best_ncomp

## ----chunk-027----------------------------------------------------
data.frame(
    ncomp = cv_opt$ncomp,
    training_R2Y = round(cv_opt$R2Y, 3),
    heldout_Q2Y = round(cv_opt$Q2Y, 3),
    heldout_RMSD = round(cv_opt$RMSD, 3)
)

## ----chunk-028----------------------------------------------------
perm_fit <- pls(
    Xtrain = Xreg_train,
    Ytrain = Ytrain_reg,
    Xtest = Xreg_test,
    Ytest = Ytest_reg,
    ncomp = 2,
    fit = TRUE,
    perm.test = TRUE,
    return_variance = FALSE,
    seed = 108
)

perm_fit$pval
plot.permutation(perm_fit, ncomp = 2)

## ----chunk-029----------------------------------------------------
dcv_perm <- pls.double.cv(
    Xdata = Xreg_train[1:20, ],
    Ydata = Ytrain_reg[1:20],
    ncomp = 1:2,
    kfold_inner = 2,
    kfold_outer = 2,
    perm.test = TRUE,
    seed = 109
)

data.frame(
    permutation_metric = dcv_perm$permutation_metric,
    observed = dcv_perm$permutation_observed,
    p_value = dcv_perm$p.value,
    completed = dcv_perm$permutation_completed,
    failed = dcv_perm$permutation_failed
)

## ----balanced-permutation-example, eval=FALSE---------------------
# dcv_balanced <- pls.double.cv(
#     Xdata = Xtrain,
#     Ydata = Ytrain_cls,
#     ncomp = 1:5,
#     kfold_inner = 5,
#     kfold_outer = 5,
#     constrain = patient_id,
#     selection = "balanced_accuracy",
#     perm.test = TRUE,
#     times = 100,
#     seed = 109
# )
# dcv_balanced$balanced_accuracy
# dcv_balanced$metrics$permutation

## ----chunk-030----------------------------------------------------
cv_select <- pls.single.cv(
    Xdata = Xtrain,
    Ydata = Ytrain_cls,
    ncomp = 1:3,
    kfold = 5,
    seed = 106
)

selected <- utils::modifyList(
    cv_select$tuning_config,
    cv_select$best_parameters
)
svd_controls <- selected$svd_dots
selected$svd_dots <- NULL
fit_selected <- do.call(
    pls,
    c(
        list(
            Xtrain = Xtrain,
            Ytrain = Ytrain_cls,
            Xtest = Xtest,
            Ytest = Ytest_cls,
            return_variance = FALSE
        ),
        selected,
        svd_controls
    )
)

data.frame(
    best_ncomp = cv_select$best_ncomp,
    test_accuracy = mean(fit_selected$Ypred[[1]] == Ytest_cls)
)

## ----chunk-031----------------------------------------------------
cv_kernel <- pls.single.cv(
    Xdata = Xtrain,
    Ydata = Ytrain_cls,
    ncomp = 1:3,
    kfold = 5,
    method = "kernelpls",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 107
)

selected_kernel <- utils::modifyList(
    cv_kernel$tuning_config,
    cv_kernel$best_parameters
)
svd_controls <- selected_kernel$svd_dots
selected_kernel$svd_dots <- NULL
fit_kernel <- do.call(
    pls,
    c(
        list(
            Xtrain = Xtrain,
            Ytrain = Ytrain_cls,
            Xtest = Xtest,
            Ytest = Ytest_cls,
            return_variance = FALSE
        ),
        selected_kernel,
        svd_controls
    )
)

data.frame(
    best_ncomp = cv_kernel$best_parameters$ncomp,
    best_kernel = cv_kernel$best_parameters$kernel,
    test_accuracy = mean(fit_kernel$Ypred[[1]] == Ytest_cls)
)

## ----chunk-032----------------------------------------------------
cv_double <- pls.double.cv(
    Xdata = Xtrain,
    Ydata = Ytrain_cls,
    constrain = patient_id,
    ncomp = 1:3,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls",
    classifier = "lda",
    selection = "balanced_accuracy",
    backend = "cpu",
    return_splits = TRUE,
    seed = 104
)

data.frame(
    selected_ncomp_mode = cv_double$bcomp,
    outer_metric = cv_double$metric_name,
    outer_accuracy = cv_double$accuracy,
    outer_balanced_accuracy = cv_double$balanced_accuracy,
    outer_Q2Y = cv_double$Q2Y,
    outer_R2Y = cv_double$R2Y
)

data.frame(
    outer_fold = seq_along(cv_double$results[[1]]$best_ncomp),
    inner_selected_ncomp = cv_double$results[[1]]$best_ncomp
)

head(cv_double$split_index)

## ----nested-regression-examples-----------------------------------
cv_double_uni <- pls.double.cv(
    Xdata = Xreg_train,
    Ydata = Ytrain_reg,
    ncomp = 1:2,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "simpls",
    backend = "cpu",
    seed = 105
)

cv_double_multi <- pls.double.cv(
    Xdata = Xmulti_train,
    Ydata = Ymulti_train,
    ncomp = 1:2,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "plssvd",
    backend = "cpu",
    bycol = TRUE,
    seed = 105
)

data.frame(
    task = c("univariate regression", "multivariate regression"),
    selected_ncomp = c(cv_double_uni$bcomp, cv_double_multi$bcomp),
    outer_RMSD = c(cv_double_uni$RMSD, cv_double_multi$RMSD)
)

## ----chunk-033----------------------------------------------------
cv_double_kernel <- pls.double.cv(
    Xdata = Xtrain,
    Ydata = Ytrain_cls,
    constrain = patient_id,
    ncomp = 1:2,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "kernelpls",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 108
)

cv_double_kernel$results[[1]]$best_parameters

## ----chunk-034----------------------------------------------------
s64 <- fastsvd(Xtrain, ncomp = 3, seed = 104)
s32 <- fastsvd(float::fl(Xtrain), ncomp = 3, seed = 104)
names(s64)
s64$d
inherits(s32$u, "float32")

## ----chunk-038----------------------------------------------------
has_cuda()
has_metal()

## ----chunk-039, eval = FALSE--------------------------------------
# if (has_cuda()) {
#     fit_gpu <- pls(
#     Xtrain,
#     Ytrain_cls,
#     Xtest,
#     Ytest_cls,
#     ncomp = 1:2,
#     backend = "cuda"
#     )
# 
#     fit_gpu_lda <- pls(
#     Xtrain,
#     Ytrain_cls,
#     Xtest,
#     Ytest_cls,
#     ncomp = 1:2,
#     method = "plssvd",
#     backend = "cuda",
#     classifier = "lda"
#     )
# }

## ----chunk-040, eval = FALSE--------------------------------------
# if (has_metal()) {
#     Xtrain_metal <- float::fl(as.matrix(Xtrain))
#     Xtest_metal <- float::fl(as.matrix(Xtest))
#     fit_metal <- pls(
#         Xtrain_metal,
#         Ytrain_cls,
#         Xtest_metal,
#         Ytest_cls,
#         ncomp = 1:2,
#         backend = "metal"
#     )
# }

## ----chunk-041----------------------------------------------------
C <- fastcor(Xtrain, byrow = FALSE, diag = FALSE)
dim(C)

## ----chunk-042----------------------------------------------------
vip <- ViP(fit_reg)
dim(vip)

## ----session-info-------------------------------------------------
sessionInfo()

