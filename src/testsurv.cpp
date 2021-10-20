#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]


arma::uword findstart2(arma::vec& y,
                       arma::uvec& start,
                       arma::uvec& stop)
{
  arma::uword n = y.n_elem;
  
  arma::uword j = 1;
  for(arma::uword i = 0; i < n-1; i++)
  {
    if(y(i)==y(i+1))
    {
      
    }else{
      start(j) = i+1;
      stop(j-1) = i;
      j++;
    }
  }
  stop(j-1) = n-1;
  
  start.resize(j);
  stop.resize(j);
  
  return j;
}



double logrank(arma::uvec& ind,
               arma::vec& y,
               arma::uvec& dlt)
{
  arma::uword n = y.n_elem;
  
  arma::uvec start(n), stop(n);
  start(0) = 0;
  arma::uword k = findstart2(y, start, stop);
  arma::vec yu = y(start);
  
  arma::uvec dlt1 = dlt % ind;
  arma::vec y1 = y(arma::find(ind == 1));
  arma::vec len(k), len1(k), r1(k), r(k);
  
  
  for(arma::uword i = 0; i<k; i++)
  {
    len(i) = arma::sum(dlt( arma::regspace<arma::uvec>(start(i),stop(i)) ));
    len1(i) = arma::sum(dlt1( arma::regspace<arma::uvec>(start(i),stop(i)) ));
    r1(i) = arma::sum( y1 >= yu(i));
    r(i) = arma::sum( y >= yu(i));
  }
  
  double nu = -arma::sum(r%len1 - r1%len);
  return nu/n/n;
}

//' Test for censored data (same covariates in f and g). 
//' 
//' The functions f and g use the same pre-specified set of covariates 
//' and are estimated from working AFT models. Note that the follow-up time is on the log scale.
//' 
//' @param x A numeric matrix of covariates to be used in f and g. 
//' @param xu A matrix of partitions: each column corresponds to a binary partition and each row 
//' corresponds to an observation. 
//' The elements in the matrix must be 1 (group 1) or 2 (group 2). 
//' The element in the ith row and jth column indicates the group membership of the ith observation
//' based on the jth partition.
//' @param y A numeric vector containing the follow-up time on the log scale.
//' @param dlt A binary vector containing the event indicator: 1 - event, 0 - censored.
//' @param tr A binary vector containing the treatment indicator: 1 - treatment, 0 - control.
//' @param B An integer of the number of bootstrap samples
//' @return The p value of the test.
//' @examples
//' set.seed(2)
//' N <- 800
//' beta <- rep(0,length = 50)
//' beta[1] <- 1
//' beta[50] <- -1
//' X <- matrix(rnorm(N*50), ncol = 50)
//' TR <- rbinom(N, 1, 0.5)
//' Y0 <-  X %*% beta + TR*0.5*(2*(X[,1]>0 | X[,25]>0)-1)+ rnorm(N, sd = 0.5)
//' Y0 <- as.vector(Y0)
//' C <- log(rexp(N, 0.1))
//' Y <- pmin(Y0,C)
//' DLT <- Y0 <= C
//' X2 <- apply(X,2,function(u) (u<0)+1)
//' # p-value
//' subgroupTEtestSurv(X[,c(1,25,50)], X2, Y, DLT, TR, 1000)
//' subgroupTEtestSurv(X[,c(1,50)], X2, Y, DLT, TR, 1000)
//' @export
// [[Rcpp::export]]
double subgroupTEtestSurv(arma::mat& x,
                          arma::umat& xu,
                          arma::vec& y,
                          arma::uvec& dlt,
                          arma::uvec& tr,
                          arma::uword B)
{
  arma::uword P = xu.n_cols;
  arma::uword P2 = x.n_cols;
  arma::uword n = xu.n_rows;
  
  arma::vec betal(P);
  arma::vec betar(P);
  
  
  Environment pkg = Environment::namespace_env("faft");
  Function f = pkg["faft"];
  
  List fit = f( Named("dlt", dlt), Named("x", y), Named("z", x) );
  
  arma::vec coef = fit[1];
  arma::vec y0 = y - x*coef;
  
  arma::uvec ord = arma::sort_index(y0);
  y0 = y0(ord);
  y = y(ord);
  dlt = dlt(ord);
  tr = tr(ord);
  x = x.rows(ord);
  xu = xu.rows(ord);
  
  for(arma::uword p = 0; p < P; p++)
  {
    arma::uvec xup = xu.col(p);
    arma::uvec indx1 = arma::find(xup == 1);
    arma::uvec indx2 = arma::find(xup == 2);
    
    arma::vec y1 = y0( indx1 );
    arma::uvec d1 = dlt( indx1 );
    arma::uvec tr1 = tr( indx1 );
    arma::vec y2 = y0( indx2 );
    arma::uvec d2 = dlt( indx2 );
    arma::uvec tr2 = tr( indx2 );
    
    betar(p) = logrank(tr1,y1,d1);
    betal(p) = logrank(tr2,y2,d2);
  }
  
  double betao = logrank(tr,y0,dlt);
  
  
  
  arma::mat trm(n,P2+1);
  
  // X + X*TR + TR
  trm.col(P2) = arma::conv_to<arma::vec>::from(tr);
  for(arma::uword p = 0; p < P2; p++)
  {
    trm.col(p) = tr % x.col(p);
  }
  arma::mat trx = arma::join_horiz<arma::mat>(x, trm);
  
  List fit3 = f( Named("dlt", dlt), Named("x", y), Named("z", trx) );
  
  arma::vec coef3 = fit3[1];
  coef3(arma::regspace<arma::uvec>(0,P2-1)).zeros();
  arma::vec e = y - trx*coef3;
  arma::vec g = trx*coef3;
  
  List fit4 = f( Named("dlt", dlt), Named("x", e), Named("z", x) );
  
  arma::vec coef4 = fit4[1];
  arma::vec e0 = e - x*coef4;
  
  arma::vec beta2r(P);
  arma::vec beta2l(P);
  
  arma::uvec ord2 = arma::sort_index(e0);
  e0 = e0(ord2);
  e = e(ord2);
  dlt = dlt(ord2);
  tr = tr(ord2);
  x = x.rows(ord2);
  xu = xu.rows(ord2);
  
  for(arma::uword p = 0; p < P; p++)
  {
    arma::uvec xup = xu.col(p);
    arma::uvec indx1 = arma::find(xup == 1);
    arma::uvec indx2 = arma::find(xup == 2);
    
    arma::vec e1 = e0( indx1 );
    arma::uvec d1 = dlt( indx1 );
    arma::uvec tr1 = tr( indx1 );
    arma::vec e2 = e0( indx2 );
    arma::uvec d2 = dlt( indx2 );
    arma::uvec tr2 = tr( indx2 );
    
    beta2r(p) = logrank(tr1,e1,d1);
    beta2l(p) = logrank(tr2,e2,d2);
  }
  
  double beta2o = logrank(tr,e0,dlt);
  
  
  
  
  arma::mat thetabr(P,B);
  arma::mat thetabl(P,B);
  arma::vec thetabo(B);
  for(arma::uword b = 0; b < B; b++)
  {
    arma::uvec indb = arma::randi<arma::uvec>(n,arma::distr_param(0,n-1));
    arma::vec  yb = e(indb);
    arma::uvec trb = tr(indb);
    arma::uvec db = dlt(indb);
    arma::vec betabr(P);
    arma::vec betabl(P);
    arma::mat xb = x.rows(indb);
    arma::umat xub = xu.rows(indb);
    
    List fitb = f( Named("dlt", db), Named("x", yb), Named("z", xb) );
    
    arma::vec coefb = fitb[1];
    
    yb = yb - xb*coefb;
    
    arma::uvec ordb = arma::sort_index(yb);
    yb = yb(ordb);
    db = db(ordb);
    trb = trb(ordb);
    xb = xb.rows(ordb);
    xub = xub.rows(ordb);
    
    for(arma::uword p = 0; p < P; p++)
    {
      arma::uvec xubp = xub.col(p); 
      arma::uvec indx1 = arma::find(xubp == 1);
      arma::uvec indx2 = arma::find(xubp == 2);
      arma::vec y1 = yb( indx1 );
      arma::uvec d1 = db( indx1 );
      arma::uvec tr1 = trb( indx1 );
      arma::vec y2 = yb( indx2 );
      arma::uvec d2 = db( indx2 );
      arma::uvec tr2 = trb( indx2 );
      
      betabr(p) = logrank(tr1,y1,d1);
      betabl(p) = logrank(tr2,y2,d2);
    }
    
    thetabl.col(b) = betabl - beta2l;
    thetabr.col(b) = betabr - beta2r;
    
    double betabo = logrank(trb,yb,db);
    thetabo(b) = betabo - beta2o;
  }
  
  arma::vec sdlP(P);
  arma::vec sdrP(P);
  arma::mat thetalsd(P,B);
  arma::mat thetarsd(P,B);
  
  for(arma::uword p = 0; p<P; p++)
  {
    sdrP(p) = arma::stddev(thetabr.row(p));
    sdlP(p) = arma::stddev(thetabl.row(p));
    thetarsd.row(p) = thetabr.row(p)/sdrP(p);
    thetalsd.row(p) = thetabl.row(p)/sdlP(p);
  }
  double sdo = arma::stddev( thetabo );
  
  arma::vec thetaBsd(B);
  for(arma::uword b = 0; b<B; b++)
  {
    thetaBsd(b) = std::max( thetabo(b)/sdo,
             std::max( arma::max(thetalsd.col(b)), arma::max(thetarsd.col(b)) ) );
  }
  
  arma::vec betarsd = betar/sdrP;
  arma::vec betalsd = betal/sdlP;
  double betaosd = betao/sdo;
  
  thetaBsd = arma::sort(thetaBsd);
  
  double ts = std::max( betaosd ,
                        std::max(betarsd.max(),
                                 betalsd.max()) );
  
  return  arma::sum(thetaBsd > ts)*1.0/B;
}

//' Test for censored data (user-specified g). 
//' 
//' Users can specify the values of the g function for each observation. 
//' The f function is estimated using a working AFT model. Note that the follow-up time is on the log scale.
//' 
//' @param x A numeric matrix of covariates to be used in f.
//' @param xu A matrix of partitions: each column corresponds to a binary partition and each row 
//' corresponds to an observation. 
//' The elements in the matrix must be 1 (group 1) or 2 (group 2). 
//' The element in the ith row and jth column indicates the group membership of the ith observation
//' based on the jth partition.
//' @param y A numeric vector containing the follow-up time on the log scale.
//' @param dlt A binary vector containing the event indicator: 1 - event, 0 - censored.
//' @param tr A binary vector containing the treatment indicator: 1 - treatment, 0 - control.
//' @param g A vector containing the values of g.
//' @param B An integer of the number of bootstrap samples.
//' @return The p value of the test.
//' @examples
//' set.seed(2)
//' N <- 800
//' beta <- rep(0,length = 50)
//' beta[1] <- 1
//' beta[50] <- -1
//' X <- matrix(rnorm(N*50), ncol = 50)
//' TR <- rbinom(N, 1, 0.5)
//' Y0 <-  X %*% beta + TR*0.5*(2*(X[,1]>0 | X[,25]>0)-1)+ rnorm(N, sd = 0.5)
//' Y0 <- as.vector(Y0)
//' C <- log(rexp(N, 0.1))
//' Y <- pmin(Y0,C)
//' DLT <- Y0 <= C
//' X2 <- apply(X,2,function(u) (u<0)+1)
//' X1 <- X[,c(1,25,50)]
//' # p-value (g = 0)
//' subgroupTEtestSurv_g(X1, X2, Y, DLT, TR, rep(0,N), 1000)
//' 
//' # Estimate g from a working AFT model
//' library(faft)
//' fitg <- faft(Y, DLT, cbind(X1,TR*X1,TR))
//' G <- cbind(TR*X1,TR) %*% fitg$beta[4:7]
//' # p value
//' subgroupTEtestSurv_g(X1, X2, Y, DLT, TR, G, 1000)
//' @export
// [[Rcpp::export]]
double subgroupTEtestSurv_g(arma::mat& x,
                            arma::umat& xu,
                            arma::vec& y,
                            arma::uvec& dlt,
                            arma::uvec& tr,
                            arma::vec& g,
                            arma::uword B)
{
  arma::uword P = xu.n_cols;
  arma::uword n = xu.n_rows;
  
  arma::vec betal(P);
  arma::vec betar(P);
  
  
  Environment pkg = Environment::namespace_env("faft");
  Function f = pkg["faft"];
  
  List fit = f( Named("dlt", dlt), Named("x", y), Named("z", x) );
  
  arma::vec coef = fit[1];
  arma::vec y0 = y - x*coef;
  
  arma::uvec ord = arma::sort_index(y0);
  y0 = y0(ord);
  y = y(ord);
  dlt = dlt(ord);
  tr = tr(ord);
  x = x.rows(ord);
  xu = xu.rows(ord);
  g = g(ord);
  
  for(arma::uword p = 0; p < P; p++)
  {
    arma::uvec xup = xu.col(p);
    arma::uvec indx1 = arma::find(xup == 1);
    arma::uvec indx2 = arma::find(xup == 2);
    
    arma::vec y1 = y0( indx1 );
    arma::uvec d1 = dlt( indx1 );
    arma::uvec tr1 = tr( indx1 );
    arma::vec y2 = y0( indx2 );
    arma::uvec d2 = dlt( indx2 );
    arma::uvec tr2 = tr( indx2 );
    
    betar(p) = logrank(tr1,y1,d1);
    betal(p) = logrank(tr2,y2,d2);
  }
  
  double betao = logrank(tr,y0,dlt);
  
  arma::vec e = y - g;
  
  List fit4 = f( Named("dlt", dlt), Named("x", e), Named("z", x) );
  
  arma::vec coef4 = fit4[1];
  arma::vec e0 = e - x*coef4;
  
  arma::vec beta2r(P);
  arma::vec beta2l(P);
  
  arma::uvec ord2 = arma::sort_index(e0);
  e0 = e0(ord2);
  e = e(ord2);
  dlt = dlt(ord2);
  tr = tr(ord2);
  x = x.rows(ord2);
  xu = xu.rows(ord2);
  
  for(arma::uword p = 0; p < P; p++)
  {
    arma::uvec xup = xu.col(p);
    arma::uvec indx1 = arma::find(xup == 1);
    arma::uvec indx2 = arma::find(xup == 2);
    
    arma::vec e1 = e0( indx1 );
    arma::uvec d1 = dlt( indx1 );
    arma::uvec tr1 = tr( indx1 );
    arma::vec e2 = e0( indx2 );
    arma::uvec d2 = dlt( indx2 );
    arma::uvec tr2 = tr( indx2 );
    
    beta2r(p) = logrank(tr1,e1,d1);
    beta2l(p) = logrank(tr2,e2,d2);
  }
  
  double beta2o = logrank(tr,e0,dlt);
  
  
  
  
  arma::mat thetabr(P,B);
  arma::mat thetabl(P,B);
  arma::vec thetabo(B);
  for(arma::uword b = 0; b < B; b++)
  {
    arma::uvec indb = arma::randi<arma::uvec>(n,arma::distr_param(0,n-1));
    arma::vec  yb = e(indb);
    arma::uvec trb = tr(indb);
    arma::uvec db = dlt(indb);
    arma::vec betabr(P);
    arma::vec betabl(P);
    arma::mat xb = x.rows(indb);
    arma::umat xub = xu.rows(indb);
    
    List fitb = f( Named("dlt", db), Named("x", yb), Named("z", xb) );
    
    arma::vec coefb = fitb[1];
    
    yb = yb - xb*coefb;
    
    arma::uvec ordb = arma::sort_index(yb);
    yb = yb(ordb);
    db = db(ordb);
    trb = trb(ordb);
    xb = xb.rows(ordb);
    xub = xub.rows(ordb);
    
    for(arma::uword p = 0; p < P; p++)
    {
      arma::uvec xubp = xub.col(p); 
      
      arma::uvec indx1 = arma::find(xubp == 1);
      arma::uvec indx2 = arma::find(xubp == 2);
      arma::vec y1 = yb( indx1 );
      arma::uvec d1 = db( indx1 );
      arma::uvec tr1 = trb( indx1 );
      arma::vec y2 = yb( indx2 );
      arma::uvec d2 = db( indx2 );
      arma::uvec tr2 = trb( indx2 );
      
      betabr(p) = logrank(tr1,y1,d1);
      betabl(p) = logrank(tr2,y2,d2);
    }
    
    thetabl.col(b) = betabl - beta2l;
    thetabr.col(b) = betabr - beta2r;
    
    double betabo = logrank(trb,yb,db);
    thetabo(b) = betabo - beta2o;
  }
  
  arma::vec sdlP(P);
  arma::vec sdrP(P);
  arma::mat thetalsd(P,B);
  arma::mat thetarsd(P,B);
  
  for(arma::uword p = 0; p<P; p++)
  {
    sdrP(p) = arma::stddev(thetabr.row(p));
    sdlP(p) = arma::stddev(thetabl.row(p));
    thetarsd.row(p) = thetabr.row(p)/sdrP(p);
    thetalsd.row(p) = thetabl.row(p)/sdlP(p);
  }
  double sdo = arma::stddev( thetabo );
  
  arma::vec thetaBsd(B);
  for(arma::uword b = 0; b<B; b++)
  {
    thetaBsd(b) = std::max( thetabo(b)/sdo,
             std::max( arma::max(thetalsd.col(b)), arma::max(thetarsd.col(b)) ) );
  }
  
  arma::vec betarsd = betar/sdrP;
  arma::vec betalsd = betal/sdlP;
  double betaosd = betao/sdo;
  
  thetaBsd = arma::sort(thetaBsd);
  
  double ts = std::max( betaosd ,
                        std::max(betarsd.max(),
                                 betalsd.max()) );
  
  return  arma::sum(thetaBsd > ts)*1.0/B;
}
