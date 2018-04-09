#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "rt_functions.h"



// --------------------------------- Response Time Model Functions -------------------------------------------
// Simulating response time, computing likelihood of response times          
// -----------------------------------------------------------------------------------------------------------

// This creates the JxJ incidence matrices indicating whether (1) item j is ahead of j' and
// (2) item j and j' has requisite skill overlaps
// [[Rcpp::export]]
arma::cube J_incidence_cube(const arma::mat& test_order, const arma::cube& Qs){
  unsigned int n_versions = test_order.n_rows;
  unsigned int Jt = Qs.n_rows;
  unsigned int J = Jt * n_versions;
  unsigned int K = Qs.n_cols;
  arma::cube J_inc = arma::zeros<arma::cube>(J,J,n_versions);
  for(unsigned int version = 0; version < n_versions;version++){
    // get the q corresponding to that version
    arma::mat Q_version = arma::zeros<arma::mat>(J,K);
    for(unsigned int t =  0; t < n_versions; t++){
      unsigned int block = test_order(version,t)-1;
      Q_version.submat((t*Jt),0,((t+1)*Jt-1),(K-1)) = Qs.slice(block);
    }
    // fill in the incident matrix
    for(unsigned int j = 1; j<J; j++){
      for(unsigned int jp = 0; jp<j; jp++){
        if(arma::dot(Q_version.row(j),Q_version.row(jp))>0){
          J_inc(j,jp,version) = 1;
        }
      }
    }
  }
  return(J_inc);
}


// This is a (hopefully) more efficient algorithm for calculating G in version 2
// [[Rcpp::export]]
arma::vec G2vec_efficient(const arma::cube& ETA, const arma::cube& J_incidence, const arma::cube& alphas_i, 
                          int test_version_i, const arma::mat test_order, unsigned int t){
  unsigned int J = J_incidence.n_cols;
  unsigned int Jt = ETA.n_rows;
  unsigned int K = alphas_i.n_cols;
  arma::colvec etas_i = arma::zeros<arma::colvec>(J);
  // get the attribute vector of i up to time t
  for(unsigned int tt = 0; tt<(t+1); tt++){
    double class_itt = arma::dot(alphas_i.slice(tt), bijectionvector(K));
    unsigned int block_itt = test_order(test_version_i,tt)-1;
    etas_i.subvec((tt*Jt),((tt+1)*Jt-1)) = ETA.slice(block_itt).col(class_itt);
  }
  // get amt of practice
  arma::mat J_i = J_incidence.slice(test_version_i);
  arma::vec G = arma::conv_to<arma::vec>::from(J_i.submat((t*Jt),0,((t+1)*Jt-1),(J-1)) * etas_i);
  
  return(log(G+1.));
}



//' @title Simulate item response times based on Wang et al.'s (2018) joint model of response times and accuracy in learning
//' @description Simulate a cube of subjects' response times across time points according to a variant of the logNormal model
//' @param alphas An N-by-K-by-T \code{array} of attribute patterns of all persons across T time points 
//' @param RT_itempars A J-by-2-by-T \code{array} of item time discrimination and time intensity parameters across item blocks
//' @param Qs A J-by-K-by-T  \code{cube} of Q-matrices across all item blocks
//' @param taus A length N \code{vector} of latent speed of each person
//' @param phi A \code{scalar} of slope of increase in fluency over time due to covariates (G)
//' @param ETA A J-by-2^K-by-T \code{array} of ideal responses across all item blocks, with each slice generated with ETAmat function
//' @param G_version An \code{int} of the type of covariate for increased fluency (1: G is dichotomous depending on whether all skills required for
//' current item are mastered; 2: G cumulates practice effect on previous items using mastered skills; 3: G is a time block effect invariant across 
//' subjects with different attribute trajectories)
//' @param test_order A N_versions-by-T \code{matrix} indicating which block of items were administered to examinees with specific test version.
//' @param Test_versions A length N \code{vector} of the test version of each examinee
//' @return A \code{cube} of response times of subjects on each item across time
//' @examples
//' class_0 <- sample(1:2^K, N, replace = T)
//' Alphas_0 <- matrix(0,N,K)
//' mu_thetatau = c(0,0)
//' Sig_thetatau = rbind(c(1.8^2,.4*.5*1.8),c(.4*.5*1.8,.25))
//' Z = matrix(rnorm(N*2),N,2)
//' thetatau_true = Z%*%chol(Sig_thetatau)
//' thetas_true = thetatau_true[,1]
//' taus_true = thetatau_true[,2]
//' G_version = 3
//' phi_true = 0.8
//' for(i in 1:N){
//'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
//' }
//' lambdas_true <- c(-2, .4, .055)     
//' Alphas <- simulate_alphas_HO_joint(lambdas_true,thetas_true,Alphas_0,Q_examinee,T,Jt)
//' RT_itempars_true <- array(NA, dim = c(Jt,2,T))
//' RT_itempars_true[,2,] <- rnorm(Jt*T,3.45,.5)
//' RT_itempars_true[,1,] <- runif(Jt*T,1.5,2)
//' ETAs <- array(NA,dim = c(Jt,2^K,T)) 
//' for(t in 1:T){
//'   ETAs[,,t] <- ETAmat(K,Jt,Q_list[[t]])
//' }
//' L_sim <- sim_RT(Alphas,RT_itempars_true,Qs,taus_true,phi_true,ETAs,
//' G_version,test_order,Test_versions)
//' @export
// [[Rcpp::export]]
arma::cube sim_RT(const arma::cube& alphas, const arma::cube& RT_itempars, const arma::cube& Qs,
                  const arma::vec& taus, double phi, const arma::cube ETA, int G_version,
                  const arma::mat& test_order, arma::vec Test_versions){
  unsigned int N = alphas.n_rows;
  unsigned int Jt = RT_itempars.n_rows;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::cube J_incidence = J_incidence_cube(test_order,Qs);
  arma::uvec practice_items;
  arma::vec vv = bijectionvector(K);
  arma::vec G(Jt);
  arma::cube L(N,Jt,T);
  for(unsigned int i = 0; i<N; i++){                       
    int test_version_i = Test_versions(i)-1;
    double tau_i= taus(i);
    for(unsigned int t = 0; t<T; t++){               
      int test_block_it = test_order(test_version_i,t)-1;
      double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
      if(G_version == 1){
        G = ETA.slice(test_block_it).col(class_it);
      }
      if(G_version == 2){
        G = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                            test_order,t);
      }
      if(G_version==3){
        G=G.fill((t+1.)/T);
      }
      
      for(unsigned int j = 0; j<Jt; j++){                    
        double lm = RT_itempars(j,1,test_block_it)-tau_i-phi*G(j);
        double ls = 1./RT_itempars(j,0,test_block_it);
        L(i,j,t) = R::rlnorm(lm,ls);
      }
    }
  }
  
  return(L);
}


// likelihood of response time
// G_it: vector of the Gs of subject i at time t
// L_it: vector of the latencies of i at time t
// RT_itempars_it: response time item parameters (gamma and a) of all items at time t for i
// [[Rcpp::export]]
double dLit(const arma::vec& G_it, const arma::vec& L_it, const arma::mat& RT_itempars_it, 
            double tau_i, double phi){
  unsigned int Jt = L_it.n_elem;
  arma::vec ps(Jt);
  double lm, ls;
  for(unsigned int j = 0; j<Jt; j++){
    lm = RT_itempars_it(j,1)-tau_i-phi*G_it(j);
    ls = 1./RT_itempars_it(j,0);
    ps(j) = R::dlnorm(L_it(j),lm,ls,0);
  }
  return(arma::prod(ps));
}