#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

//' Test for uncensored data
//' 
//' 
//' @param xu A matrix of partitions: each column corresponds to a binary partition and each row 
//' corresponds to an observation. 
//' The elements in the matrix must be 1 (group 1) or 2 (group 2). 
//' The element in the ith row and jth column indicates the group membership of the ith observation
//' based on the jth partition.
//' @param y A numeric vector containing the outcome.
//' @param tr A binary vector containing the treatment indicator: 1 - treatment, 0 - control.
//' @param fx A numeric matrix of covariates to be used in f.
//' @param g A vector containing the values of g.
//' @param B An integer of the number of bootstrap samples.
//' @return The p value of the test.
//' @examples
//' set.seed(1)
//' N <- 800
//' beta <- rep(0,length = 50)
//' beta[1] <- 1
//' beta[50] <- -1
//' X <- matrix(rnorm(N*50), ncol = 50)
//' TR <- rbinom(N, 1, 0.5)
//' Y <-  X %*% beta + TR*0.5*(2*(X[,1]>0 | X[,25]>0)-1)+ rnorm(N, sd = 0.5)
//'   
//' # partition matrix
//' X2 <- apply(X,2,function(u) (u<0)+1)
//' # covariates in f
//' X1 <- X[,c(1,25,50)]
//' # p-value (g = 0)
//' subgroupTEtest(X2,Y,TR,X1,rep(0,N),1000)
//' 
//' # estimate g from a working linear model
//' fitg <- lm(Y~X1*TR)
//' G <- cbind(TR,TR*X1) %*% coef(fitg)[5:8]
//' subgroupTEtest(X2,Y,TR,X1,G,1000)
//' @export
// [[Rcpp::export]]
double subgroupTEtest(arma::umat& xu, 
                      arma::vec& y,
                      arma::uvec& tr,
                      arma::mat& fx,  
                      arma::mat& g,
                      arma::uword B)   
{
  arma::uword P = xu.n_cols;
  arma::uword n = xu.n_rows;
  
  arma::vec betal(P);
  arma::vec betar(P);
  
  arma::mat xone(n,1);
  xone.ones();
  arma::mat fx1 = arma::join_horiz<arma::mat>(xone,fx);
  arma::vec coef = arma::solve(fx1, y);  
  arma::vec y0 = y - fx1*coef; 
  
  
  for(arma::uword p = 0; p < P; p++)
  {
    arma::uvec xup = xu.col(p);
    arma::vec y1 = y0( arma::find(xup == 1) );
    arma::uvec tr1 = tr( arma::find(xup == 1) );
    arma::vec y2 = y0( arma::find(xup == 2) );
    arma::uvec tr2 = tr( arma::find(xup == 2) );
    betar(p) = (arma::sum( y1 % tr1 )*arma::sum( 1-tr1 ) - arma::sum( y1 % (1-tr1) )*arma::sum( tr1 ))/n/n;
    betal(p) = (arma::sum( y2 % tr2 )*arma::sum( 1-tr2 ) - arma::sum( y2 % (1-tr2) )*arma::sum( tr2 ))/n/n;
  }
  
  double betao = (arma::sum( y0 % tr )*arma::sum( 1-tr ) - arma::sum( y0 % (1-tr) )*arma::sum( tr ))/n/n;
  
  
  arma::vec e = y - g; 
  
  arma::vec beta2r(P);
  arma::vec beta2l(P);
  
  arma::vec coef2 = arma::solve(fx1, e);  
  arma::vec e0 = e - fx1*coef2; 
  
  for(arma::uword p = 0; p < P; p++)
  {
    arma::uvec xup = xu.col(p);
    arma::vec y1 = e0( arma::find(xup == 1) );
    arma::uvec tr1 = tr( arma::find(xup == 1) );
    arma::vec y2 = e0( arma::find(xup == 2) );
    arma::uvec tr2 = tr( arma::find(xup == 2) );
    beta2r(p) = (arma::sum( y1 % tr1 )*arma::sum( 1-tr1 ) - arma::sum( y1 % (1-tr1) )*arma::sum( tr1 ))/n/n;
    beta2l(p) = (arma::sum( y2 % tr2 )*arma::sum( 1-tr2 ) - arma::sum( y2 % (1-tr2) )*arma::sum( tr2 ))/n/n;
  }
  
  double beta2o = (arma::sum( e0 % tr )*arma::sum( 1-tr ) - arma::sum( e0 % (1-tr) )*arma::sum( tr ))/n/n;
  
  
  arma::mat thetabr(P,B);
  arma::mat thetabl(P,B);
  arma::vec thetabo(B);
  for(arma::uword b = 0; b < B; b++)
  {
    arma::uvec indb = arma::randi<arma::uvec>(n,arma::distr_param(0,n-1));
    arma::vec  yb = e(indb);
    arma::uvec trb = tr(indb);
    arma::vec betabr(P);
    arma::vec betabl(P);
    arma::mat fxb = fx.rows(indb);
    arma::umat xub = xu.rows(indb);
    
    
    arma::mat fx1b = arma::join_horiz<arma::mat>(xone,fxb);
    arma::vec coef3 = arma::solve(fx1b, yb);  
    yb = yb - fx1b*coef3; 
    
    for(arma::uword p = 0; p < P; p++)
    {
      arma::uvec xubp = xub.col(p); 
      arma::vec y1b = yb( arma::find(xubp == 1) );
      arma::uvec tr1b = trb( arma::find(xubp == 1) );
      arma::vec y2b = yb( arma::find(xubp == 2) );
      arma::uvec tr2b = trb( arma::find(xubp == 2) );
      betabr(p) = (arma::sum( y1b % tr1b )*arma::sum( 1-tr1b ) - arma::sum( y1b % (1-tr1b) )*arma::sum( tr1b ))/n/n; 
      betabl(p) = (arma::sum( y2b % tr2b )*arma::sum( 1-tr2b ) - arma::sum( y2b % (1-tr2b) )*arma::sum( tr2b ))/n/n;    
    }
    
    thetabl.col(b) = betabl - beta2l;
    thetabr.col(b) = betabr - beta2r;
    thetabo(b) = (arma::sum( yb % trb )*arma::sum( 1-trb ) - arma::sum( yb % (1-trb ) )*arma::sum( trb ))/n/n - beta2o;
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
