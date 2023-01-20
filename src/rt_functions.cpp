#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "rt_functions.h"



// --------------------------------- Response Time Model Functions -------------------------------------------
// Simulating response time, computing likelihood of response times          
// -----------------------------------------------------------------------------------------------------------


// This creates the JxJ incidence matrices indicating whether (1) item j is ahead of j' and
// (2) item j and j' has requisite skill overlaps for each examinee's Q_matrix
// [[Rcpp::export]]
arma::cube J_incidence_cube_g(const arma::mat& Q_matrix, const arma::cube& Design_array){
  unsigned int J = Q_matrix.n_rows;
  unsigned int N = Design_array.n_rows;
  
  Rcpp::List Q_examinee = Q_list_g(Q_matrix, Design_array);
  arma::cube J_inc = arma::zeros<arma::cube>(J,J,N);
  for(unsigned int i = 0; i < N;i++){
    // get the q corresponding to that examinee
    arma::mat Q_i = Q_examinee(i);
    // fill in the incident matrix
    for(unsigned int j = 1; j<J; j++){
      for(unsigned int jp = 0; jp<j; jp++){
        if(arma::dot(Q_i.row(j),Q_i.row(jp))>0){
          J_inc(j,jp,i) = 1;
        }
      }
    }
  }
  return(J_inc);
}


// This is a (hopefully) more efficient algorithm for calculating G in version 2
// [[Rcpp::export]]
arma::vec G2vec_efficient_g(const arma::mat& ETA, const arma::cube& J_incidence, const arma::cube& alphas_i, 
                          unsigned int t, const arma::mat& Q_matrix, const arma::cube& Design_array, unsigned int i){
  unsigned int J = ETA.n_rows;
  unsigned int K = alphas_i.n_cols;
  arma::colvec etas_i = arma::zeros<arma::colvec>(J);
  // get the attribute vector of i up to time t
  for(unsigned int tt = 0; tt<(t+1); tt++){
    double class_itt = arma::dot(alphas_i.slice(tt), bijectionvector(K));
    arma::uvec block_itt = arma::find(Design_array.slice(tt).row(i) == 1);
    // unsigned int block_itt = Test_order(test_version_i,tt)-1;
    arma::vec etas_tt = ETA.col(class_itt);
    etas_i(block_itt) = etas_tt(block_itt);
  }
  // get amt of practice
  arma::mat J_i = J_incidence.slice(i);
  arma::uvec block_it = arma::find(Design_array.slice(t).row(i) == 1);
  arma::vec G = arma::conv_to<arma::vec>::from(J_i.rows(block_it) * etas_i);
  // arma::vec G = arma::conv_to<arma::vec>::from(J_i.submat((t*Jt),0,((t+1)*Jt-1),(J-1)) * etas_i);
  
  return(log(G+1.));
}



//' @title Simulate item response times based on Wang et al.'s (2018) joint model of response times and accuracy in learning
//' @description Simulate a cube of subjects' response times across time points according to a variant of the logNormal model
//' @param alphas An N-by-K-by-T \code{array} of attribute patterns of all persons across T time points 
//' @param Q_matrix A J-by-K  Q-matrix for the test
//' @param Design_array A N-by-J-by-L array indicating whether item j is administered to examinee i at l time point.
//' @param RT_itempars A J-by-2 \code{matrix} of item time discrimination and time intensity parameters
//' @param taus A length N \code{vector} of latent speed of each person
//' @param phi A \code{scalar} of slope of increase in fluency over time due to covariates (G)
//' @param G_version An \code{int} of the type of covariate for increased fluency (1: G is dichotomous depending on whether all skills required for
//' current item are mastered; 2: G cumulates practice effect on previous items using mastered skills; 3: G is a time block effect invariant across 
//' subjects with different attribute trajectories)
//' @return A \code{cube} of response times of subjects on each item across time
//' @examples
//' N = dim(Design_array)[1]
//' J = nrow(Q_matrix)
//' K = ncol(Q_matrix)
//' L = dim(Design_array)[3]
//' class_0 <- sample(1:2^K, N, replace = TRUE)
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
//' Alphas <- sim_alphas(model="HO_joint", 
//'                      lambdas=lambdas_true, 
//'                      thetas=thetas_true, 
//'                      Q_matrix=Q_matrix, 
//'                      Design_array=Design_array)
//' RT_itempars_true <- matrix(NA, nrow=J, ncol=2)
//' RT_itempars_true[,2] <- rnorm(J,3.45,.5)
//' RT_itempars_true[,1] <- runif(J,1.5,2)
//' ETAs <- ETAmat(K,J,Q_matrix)
//' L_sim <- sim_RT(Alphas,Q_matrix,Design_array,RT_itempars_true,taus_true,phi_true,G_version)
//' @export
// [[Rcpp::export]]
arma::cube sim_RT(const arma::cube& alphas, const arma::mat& Q_matrix, const arma::cube& Design_array, 
                    const arma::mat& RT_itempars, const arma::vec& taus, double phi, int G_version){
  unsigned int N = alphas.n_rows;
  unsigned int J = Design_array.n_cols;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::mat ETAs = ETAmat(K,J,Q_matrix);
  arma::cube J_incidence = J_incidence_cube_g(Q_matrix, Design_array);
  arma::uvec practice_items;
  arma::vec vv = bijectionvector(K);
  arma::cube L(N,J,T, arma::fill::value(NA_REAL));
  arma::uvec block_it;
  
  for(unsigned int i = 0; i<N; i++){                       
    double tau_i= taus(i);
    for(unsigned int t = 0; t<T; t++){
      double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
      block_it = arma::find(Design_array.slice(t).row(i) == 1);
      unsigned int Jt = block_it.n_elem;
      arma::vec G(Jt);

      if(G_version == 1){
        arma::vec G_it = ETAs.col(class_it);
        G = G_it.elem(block_it);
      }
      if(G_version == 2){
        G = G2vec_efficient_g(ETAs,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),
                              t,Q_matrix, Design_array, i);
      }
      if(G_version==3){
        G=G.fill((t+1.)/T);
      }
      
      arma::vec RT_itempars_lm = RT_itempars.col(1);
      arma::vec RT_itempars_ls = RT_itempars.col(0);
      
      for(unsigned int j = 0; j<Jt; j++){
        double lm = RT_itempars_lm(block_it(j))-tau_i-phi*G(j);
        double ls = 1./RT_itempars_ls(block_it(j));
        L(i,block_it(j),t) = R::rlnorm(lm,ls);
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