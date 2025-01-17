# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

pls_light <- function(Xtrain, Ytrain, Xtest, ncomp) {
  .Call('fastPLS_pls_light',PACKAGE = 'fastPLS', Xtrain, Ytrain, Xtest, ncomp)
}

RQ <- function(yData, yPred) {
  .Call('fastPLS_RQ', PACKAGE = 'fastPLS', yData, yPred)
}

ORTHOG <- function(X, Y, T, xm, xn, yn) {
  .Call('fastPLS_ORTHOG',PACKAGE = 'fastPLS', X, Y, T, xm, xn, yn)
}

IRLB <- function(X, nu, work, maxit = 1000L, tol = 1e-5, eps = 1e-9, svtol = 1e-5) {
  .Call('fastPLS_IRLB',PACKAGE = 'fastPLS', X, nu, work, maxit, tol, eps, svtol)
}

transformy <- function(y) {
  .Call('fastPLS_transformy', PACKAGE = 'fastPLS', y)
}




pls.model1 =
  function (Xtrain, 
            Ytrain, 
            ncomp,
            fit = FALSE,
            scaling=1,
            svd.method=1) 
  {
    
    model = .Call('fastPLS_pls_model1', PACKAGE = 'fastPLS',Xtrain, Ytrain, ncomp, scaling, fit,svd.method)
    class(model)="fastPLS"
    model
  }



pls.model2 =
  function (Xtrain, 
            Ytrain, 
            ncomp,
            fit = FALSE,
            scaling=1,
            svd.method=1) 
  {
    
    model = .Call('fastPLS_pls_model2', PACKAGE = 'fastPLS',Xtrain, Ytrain, ncomp, scaling, fit,svd.method)
    class(model)="fastPLS"
    model
  }


pls_predict =
  function (model,Xtest,
            proj = FALSE) 
  {
    Xtest = as.matrix(Xtest)
    o = .Call('fastPLS_pls_predict', PACKAGE = 'fastPLS', model, Xtest, proj)
    o
  }


optim_pls_cv <- function(XData, YData, ncomp, constrain,  scal,kfold, method,svd.method) 
{

  .Call('fastPLS_optim_pls_cv', PACKAGE = 'fastPLS', XData, YData, constrain, ncomp,scal,kfold, method,svd.method)
}


double_pls_cv <- function(Xdata, Ydata, ncomp,constrain,scaling, kfold_inner, kfold_outer, method,svd.method) {
  .Call('fastPLS_double_pls_cv', PACKAGE = 'fastPLS', Xdata, Ydata, ncomp,constrain,scaling, kfold_inner, kfold_outer, method,svd.method)
}
