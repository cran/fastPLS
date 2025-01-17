// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "../inst/include/fastPLS.h"

using namespace Rcpp;

// RQ
double RQ(arma::mat yData,arma::mat yPred);
RcppExport SEXP fastPLS_RQ(SEXP yDataSEXP,SEXP yPredSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type yData(yDataSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type yPred(yPredSEXP);
  __result = Rcpp::wrap(RQ(yData,yPred));
  return __result;
  END_RCPP
}

arma::mat pls_light(arma::mat  Xtrain,arma::mat Ytrain,arma::mat Xtest,int ncomp);
RcppExport SEXP fastPLS_pls_light(SEXP XtrainSEXP,SEXP YtrainSEXP,SEXP XtestSEXP,SEXP ncompSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter< arma::mat >::type Xtrain(XtrainSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Ytrain(YtrainSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Xtest(XtestSEXP);
  Rcpp::traits::input_parameter< int >::type ncomp(ncompSEXP);
  return Rcpp::wrap(pls_light(Xtrain, Ytrain, Xtest, ncomp));
  END_RCPP
}

// ORTHOG
arma::mat ORTHOG(arma::mat& X, arma::mat& Y, arma::mat& T, int xm, int xn, int yn);
RcppExport SEXP fastPLS_ORTHOG(SEXP XSEXP, SEXP YSEXP, SEXP TSEXP, SEXP xmSEXP, SEXP xnSEXP, SEXP ynSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type T(TSEXP);
    Rcpp::traits::input_parameter< int >::type xm(xmSEXP);
    Rcpp::traits::input_parameter< int >::type xn(xnSEXP);
    Rcpp::traits::input_parameter< int >::type yn(ynSEXP);
    rcpp_result_gen = Rcpp::wrap(ORTHOG(X, Y, T, xm, xn, yn));
    return rcpp_result_gen;
END_RCPP
}
// IRLB
List IRLB(arma::mat& X, int nu, int work, int maxit, double tol, double eps, double svtol);
RcppExport SEXP fastPLS_IRLB(SEXP XSEXP, SEXP nuSEXP, SEXP workSEXP, SEXP maxitSEXP, SEXP tolSEXP, SEXP epsSEXP, SEXP svtolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< int >::type work(workSEXP);
    Rcpp::traits::input_parameter< int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type svtol(svtolSEXP);
    rcpp_result_gen = Rcpp::wrap(IRLB(X, nu, work, maxit, tol, eps, svtol));
    return rcpp_result_gen;
END_RCPP
}


// fastPLS_irlba
List pls_model1(arma::mat Xtrain, arma::mat Ytrain, arma::ivec ncomp,int scaling,bool fit,int svd_method);
RcppExport SEXP fastPLS_pls_model1(SEXP XtrainSEXP, SEXP YtrainSEXP, SEXP ncompSEXP, SEXP scalingSEXP,SEXP fitSEXP, SEXP svd_methodSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type Xtrain(XtrainSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Ytrain(YtrainSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type ncomp(ncompSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  Rcpp::traits::input_parameter< bool >::type fit(fitSEXP);
  Rcpp::traits::input_parameter< int >::type svd_method(svd_methodSEXP);
  __result = Rcpp::wrap(pls_model1(Xtrain, Ytrain, ncomp, scaling, fit, svd_method));
  return __result;
  END_RCPP
}



// fastPLS_svd_econ
List pls_predict(List& model, arma::mat Xtest, bool proj);
RcppExport SEXP fastPLS_pls_predict(SEXP modelSEXP, SEXP XtestSEXP, SEXP projSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< List& >::type model(modelSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Xtest(XtestSEXP);
  Rcpp::traits::input_parameter< bool >::type proj(projSEXP);
  __result = Rcpp::wrap(pls_predict(model, Xtest, proj));
  return __result;
  END_RCPP
}

// fastPLS_irlba
List pls_model2(arma::mat Xtrain, arma::mat Ytrain, arma::ivec ncomp,int scaling,bool fit,int svd_method);
RcppExport SEXP fastPLS_pls_model2(SEXP XtrainSEXP, SEXP YtrainSEXP, SEXP ncompSEXP, SEXP scalingSEXP,SEXP fitSEXP, SEXP svd_methodSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type Xtrain(XtrainSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Ytrain(YtrainSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type ncomp(ncompSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  Rcpp::traits::input_parameter< bool >::type fit(fitSEXP);
  Rcpp::traits::input_parameter< int >::type svd_method(svd_methodSEXP);
  __result = Rcpp::wrap(pls_model2(Xtrain, Ytrain, ncomp, scaling, fit, svd_method));
  return __result;
  END_RCPP
}



// transformy
arma::mat transformy(arma::ivec y);
RcppExport SEXP fastPLS_transformy(SEXP ySEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::ivec >::type y(ySEXP);
  __result = Rcpp::wrap(transformy(y));
  return __result;
  END_RCPP
}


// optim_pls_cv
List optim_pls_cv(arma::mat XData, arma::mat YData, arma::ivec constrain, arma::ivec ncomp,int scaling, int kfold, int method,int svd_method);
RcppExport SEXP fastPLS_optim_pls_cv(SEXP XDataSEXP, SEXP YDataSEXP, SEXP constrainSEXP, SEXP ncompSEXP, SEXP scalingSEXP, SEXP kfoldSEXP, SEXP methodSEXP, SEXP svd_methodSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type XData(XDataSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type YData(YDataSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type ncomp(ncompSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  Rcpp::traits::input_parameter< int >::type kfold(kfoldSEXP);
  Rcpp::traits::input_parameter< int >::type method(methodSEXP);
  Rcpp::traits::input_parameter< int >::type svd_method(svd_methodSEXP);
  __result = Rcpp::wrap(optim_pls_cv(XData, YData, constrain, ncomp,scaling,kfold,method, svd_method));
  return __result;
  END_RCPP
}



// double_pls_cv
List double_pls_cv(arma::mat XData, arma::mat YData,  arma::ivec ncomp,arma::ivec constrain, int scaling, int kfold_inner, int kfold_outer, int method,int svd_method) ;
RcppExport SEXP fastPLS_double_pls_cv(SEXP XDataSEXP, SEXP YDataSEXP, SEXP ncompSEXP, SEXP constrainSEXP, SEXP scalingSEXP, SEXP kfold_innerSEXP, SEXP kfold_outerSEXP, SEXP methodSEXP, SEXP svd_methodSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type XData(XDataSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type YData(YDataSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type ncomp(ncompSEXP);
  Rcpp::traits::input_parameter< arma::ivec >::type constrain(constrainSEXP);
  Rcpp::traits::input_parameter< int >::type scaling(scalingSEXP);
  Rcpp::traits::input_parameter< int >::type kfold_inner(kfold_innerSEXP);
  Rcpp::traits::input_parameter< int >::type kfold_outer(kfold_outerSEXP);
  Rcpp::traits::input_parameter< int >::type method(methodSEXP);
  Rcpp::traits::input_parameter< int >::type svd_method(svd_methodSEXP);
  __result = Rcpp::wrap(double_pls_cv(XData, YData, ncomp,constrain, scaling,kfold_inner, kfold_outer, method, svd_method));
  return __result;
  END_RCPP
}



static const R_CallMethodDef CallEntries[] = {
 {"fastPLS_ORTHOG", (DL_FUNC) &fastPLS_ORTHOG, 6},
 {"fastPLS_IRLB", (DL_FUNC) &fastPLS_IRLB, 7},
 {NULL, NULL, 0}
};

RcppExport void R_initfastPLS(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}


