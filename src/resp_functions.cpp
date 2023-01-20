#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "resp_functions.h"

// -------------------------------- Response Model Functions -----------------------------------------------
// Simulating responses (single person and response cube),  computing likelihood of response vector                                          
// ---------------------------------------------------------------------------------------------------------


// [[Rcpp::export]]
arma::vec sim_resp_DINA(unsigned int J, unsigned int K, const arma::mat& ETA,
                        arma::vec& Svec, arma::vec& Gvec,
                        arma::vec& alpha){
  arma::vec vv = bijectionvector(K);
  arma::vec one_m_s = arma::ones<arma::vec>(J) - Svec;
  double class_it = arma::dot(alpha,vv);
  arma::vec eta_it = ETA.col(class_it);
  arma::vec us = arma::randu<arma::vec>(J);
  arma::vec one_m_eta = arma::ones<arma::vec>(J) - eta_it;
  arma::vec ps = one_m_s%eta_it + Gvec%one_m_eta;
  arma::vec compare = arma::zeros<arma::vec>(J);
  compare.elem(arma::find(ps-us > 0) ).fill(1.0);
  
  return(compare);
}



// [[Rcpp::export]]
arma::cube simDINA_g(const arma::cube& alphas, const arma::mat& itempars, const arma::mat& Q_matrix,
                   const arma::cube& Design_array){
  unsigned int N = alphas.n_rows;
  unsigned int J = itempars.n_rows;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::cube Y(N,J,T, arma::fill::value(NA_REAL));
  arma::mat ETAs = ETAmat(K,J,Q_matrix);

  arma::vec svec,gvec;
  arma::vec vv = bijectionvector(K);
  arma::uvec test_block_it;

  for(unsigned int i=0;i<N;i++){
    for(unsigned int t=0;t<T;t++){
      test_block_it = arma::find(Design_array.slice(t).row(i) == 1);
      arma::vec svec_total = itempars.col(0);
      arma::vec gvec_total = itempars.col(1);
      svec = svec_total.elem(test_block_it);
      gvec = gvec_total.elem(test_block_it);
      double Jt = arma::sum(Design_array.slice(t).row(i) == 1);
      arma::vec one_m_s = arma::ones<arma::vec>(Jt) - svec;
      double class_it = arma::dot(alphas.slice(t).row(i),vv);
      arma::vec eta_it_temp = ETAs.col(class_it);
      arma::vec eta_it = eta_it_temp.elem(test_block_it);
      arma::vec us = arma::randu<arma::vec>(Jt);
      arma::vec one_m_eta = arma::ones<arma::vec>(Jt) - eta_it;
      arma::vec ps = one_m_s%eta_it + gvec%one_m_eta;
      arma::vec compare = arma::zeros<arma::vec>(Jt);
      compare.elem(arma::find(ps-us > 0) ).fill(1.0);

      arma::vec Y_it(J, arma::fill::value(NA_REAL));
      Y_it.elem(test_block_it) = compare;
      Y.slice(t).row(i) = Y_it.t();
    }
  }
  return(Y);
}

// [[Rcpp::export]]
double pYit_DINA(const arma::vec& ETA_it,const arma::vec& Y_it, const arma::mat& itempars){
  arma::vec ss = itempars.col(0);
  arma::vec gs = itempars.col(1);
  arma::vec one_m_ss = 1. - ss;
  arma::vec one_m_gs = 1. - gs;
  arma::vec one_m_ETA_it = 1. - ETA_it;
  arma::vec one_m_Y_it = 1. - Y_it;
  
  arma::vec ps = Y_it%(one_m_ss%ETA_it + gs%one_m_ETA_it) + one_m_Y_it%(ss%ETA_it + one_m_gs%one_m_ETA_it);
  
  return arma::prod(ps);
}


// [[Rcpp::export]]
arma::vec sim_resp_rRUM(unsigned int J, unsigned int K, const arma::mat& Q,
                        const arma::mat& rstar, const arma::vec& pistar,
                        const arma::vec& alpha){
  
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj;
  double aik;
  arma::vec pmu(J);
  arma::vec Yi=arma::zeros<arma::vec>(J);
  arma::vec pi=arma::ones<arma::vec>(J);
  arma::vec ui = arma::randu<arma::vec>(J);
  for(unsigned int j=0;j<J;j++){
    arma::uvec task_ij = find(Q.row(j) == 1);
    
    for(unsigned int k = 0;k<task_ij.n_elem ;k++){
      kj = task_ij(k);
      aik = alpha(kj);
      pi(j) = ((rstar(j,kj)*(1.0-aik)+1.0*aik)*Q(j,kj)+1.0*(1.0-Q(j,kj)))*pi(j);
    }
    pi(j) = pistar(j)*pi(j);
  }
  pmu = pi - ui;
  Yi(arma::find(pmu > 0)).fill(1);
  return Yi;  
}




// [[Rcpp::export]]
arma::cube simrRUM_g(const arma::cube& alphas, const arma::mat& r_stars_mat, const arma::vec& pi_stars, 
                   const arma::mat Q_matrix, const arma::cube& Design_array){
  unsigned int N = alphas.n_rows;
  unsigned int J = pi_stars.n_elem;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::uvec test_block_it;
  
  arma::cube Y(N,J,T, arma::fill::value(NA_REAL));
  
  for(unsigned int i=0;i<N;i++){
    for(unsigned int t=0;t<T;t++){
      test_block_it = arma::find(Design_array.slice(t).row(i) == 1);
      double Jt = arma::sum(Design_array.slice(t).row(i) == 1);
      arma::mat Q_it = Q_matrix.rows(test_block_it);
      arma::mat rstar_it = r_stars_mat.rows(test_block_it);
      arma::vec pistar_it = pi_stars.elem(test_block_it);
      arma::vec alpha_it = alphas.slice(t).row(i).t();
      
      arma::vec Y_it(J, arma::fill::value(NA_REAL));
      Y_it.elem(test_block_it) = sim_resp_rRUM(Jt,K,Q_it,rstar_it,pistar_it,alpha_it).t();
      Y.slice(t).row(i) = Y_it.t();
    }
  }
  return(Y);
}


// [[Rcpp::export]]
double pYit_rRUM(const arma::vec& alpha_it, const arma::vec& Y_it, const arma::vec& pi_star_it, 
                 const arma::mat& r_star_it, const arma::mat& Q_it){
  unsigned int Jt = pi_star_it.n_elem;
  unsigned int kj;
  double aik;
  arma::vec p=arma::ones<arma::vec>(Jt);
  for(unsigned int j=0;j<Jt;j++){
    arma::uvec task_ij = find(Q_it.row(j) == 1);
    for(unsigned int k = 0;k<task_ij.n_elem ;k++){
      kj = task_ij(k);
      aik = alpha_it(kj);
      p(j) = ((r_star_it(j,kj)*(1.0-aik)+1.0*aik)*Q_it(j,kj)+1.0*(1.0-Q_it(j,kj)))*p(j);
    }
    p(j) = pi_star_it(j)*p(j);
  }
  
  arma::vec probs = p%Y_it + (1-p)%(1-Y_it);
  
  return arma::prod(probs);
}


// [[Rcpp::export]]
arma::vec sim_resp_NIDA(const unsigned int J, const unsigned int K, const arma::mat& Q,
                        const arma::vec& Svec, const arma::vec& Gvec,
                        const arma::vec& alpha){
  
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj;
  double aik;
  arma::vec pmu(J);
  arma::vec Yi=arma::zeros<arma::vec>(J);
  arma::vec pi=arma::ones<arma::vec>(J);
  arma::vec ui = arma::randu<arma::vec>(J);
  for(unsigned int j=0;j<J;j++){
    arma::uvec task_ij = find(Q.row(j) == 1);
    for(unsigned int k = 0;k<task_ij.n_elem ;k++){
      kj = task_ij(k);
      aik = alpha(kj);
      pi(j) = ((1-Svec(kj))*aik+Gvec(kj)*(1-aik))*pi(j);
    }
  }
  
  pmu = pi - ui;
  Yi(arma::find(pmu > 0)).fill(1);
  return Yi;  
}



// [[Rcpp::export]]
arma::cube simNIDA_g(const arma::cube& alphas, const arma::vec& Svec, const arma::vec& Gvec, 
                   const arma::mat Q_matrix, const arma::cube& Design_array){
  unsigned int N = alphas.n_rows;
  unsigned int J = Q_matrix.n_rows;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::uvec test_block_it;
  // unsigned int Jt = J/T;
  // arma::cube Qs = Mat2Array(Q_matrix, T);
  // arma::cube Y(N,Jt,T);
  arma::cube Y(N,J,T, arma::fill::value(NA_REAL));
  for(unsigned int i=0;i<N;i++){
    // int test_version_i = Test_versions(i)-1;
    for(unsigned int t=0;t<T;t++){
      test_block_it = arma::find(Design_array.slice(t).row(i) == 1);
      double Jt = arma::sum(Design_array.slice(t).row(i) == 1);
      // int test_block_it = Test_order(test_version_i,t)-1;
      // arma::mat Q_it = Qs.slice(test_block_it);
      arma::mat Q_it = Q_matrix.rows(test_block_it);
      arma::vec alpha_it = alphas.slice(t).row(i).t();
      // Y.slice(t).row(i) = sim_resp_NIDA(Jt,K,Q_it,Svec,Gvec,alpha_it).t();
      arma::vec Y_it(J, arma::fill::value(NA_REAL));
      Y_it.elem(test_block_it) = sim_resp_NIDA(Jt,K,Q_it,Svec,Gvec,alpha_it).t();
      Y.slice(t).row(i) = Y_it.t();
    }
  }
  return(Y);
}


// [[Rcpp::export]]
double pYit_NIDA(const arma::vec& alpha_it, const arma::vec& Y_it, const arma::vec& Svec, 
                 const arma::vec& Gvec, const arma::mat& Q_it){
  unsigned int Jt = Q_it.n_rows;
  unsigned int kj;
  double aik;
  arma::vec p=arma::ones<arma::vec>(Jt);
  for(unsigned int j=0;j<Jt;j++){
    arma::uvec task_ij = find(Q_it.row(j) == 1);
    for(unsigned int k = 0;k<task_ij.n_elem ;k++){
      kj = task_ij(k);
      aik = alpha_it(kj);
      p(j) = ((1-Svec(kj))*aik+Gvec(kj)*(1-aik))*p(j);
    }
  }
  arma::vec probs = p%Y_it + (1-p)%(1-Y_it);
  
  return arma::prod(probs);
}


//' @title Simulate responses from the specified model (entire cube)
//' @description Simulate a cube of responses from the specified model for all persons on items across all time points. 
//' Currently available models are `DINA`, `rRUM`, and `NIDA`.
//' @param model The cognitive diagnostic model under which the item responses are generated
//' @param alphas An N-by-K-by-L \code{array} of attribute patterns of all persons across L time points 
//' @param Q_matrix A J-by-K of Q-matrix
//' @param Design_array A N-by-J-by-L array indicating whether item j is administered to examinee i at l time point.
//' @param itempars A J-by-2 \code{mat} of item parameters (slipping: 1st col, guessing: 2nd col).
//' @param r_stars A J-by-K \code{mat} of item penalty parameters for missing skills.
//' @param pi_stars A length J \code{vector} of item correct response probability with all requisite skills.
//' @param Svec A length K \code{vector} of slipping probability in applying mastered skills
//' @param Gvec A length K \code{vector} of guessing probability in applying mastered skills
//' @return An \code{array} of item responses from the specified model of examinees across all time points.
//' @examples
//' \donttest{
//' ## DINA ##
//' N = nrow(Design_array)
//' J = nrow(Q_matrix)
//' thetas_true = rnorm(N, 0, 1.8)
//' lambdas_true <- c(-2, .4, .055)
//' Alphas <- sim_alphas(model="HO_joint", 
//'                     lambdas=lambdas_true, 
//'                     thetas=thetas_true, 
//'                     Q_matrix=Q_matrix, 
//'                     Design_array=Design_array)
//' itempars_true <- matrix(runif(J*2,.1,.2), ncol=2)
//' 
//' Y_sim <- sim_hmcdm(model="DINA",Alphas,Q_matrix,Design_array,
//'                    itempars=itempars_true)
//'                    
//' ## rRUM ##
//' J = nrow(Q_matrix)
//' K = ncol(Q_matrix)
//' Smats <- matrix(runif(J*K,.1,.3),c(J,K))
//' Gmats <- matrix(runif(J*K,.1,.3),c(J,K))
//' r_stars <- Gmats / (1-Smats)
//' pi_stars <- apply((1-Smats)^Q_matrix, 1, prod)
//' 
//' Y_sim <- sim_hmcdm(model="rRUM",Alphas,Q_matrix,Design_array,
//'                    r_stars=r_stars,pi_stars=pi_stars)
//' 
//' ## NIDA ##
//' K = ncol(Q_matrix)
//' Svec <- runif(K,.1,.3)
//' Gvec <- runif(K,.1,.3)
//' 
//' Y_sim <- sim_hmcdm(model="NIDA",Alphas,Q_matrix,Design_array,
//'                    Svec=Svec,Gvec=Gvec)
//' }
//' @export
// [[Rcpp::export]]
arma::cube sim_hmcdm(const std::string model,const arma::cube& alphas,const arma::mat& Q_matrix,const arma::cube& Design_array,
                     const Rcpp::Nullable<arma::mat> itempars = R_NilValue,
                     const Rcpp::Nullable<arma::mat> r_stars = R_NilValue,
                     const Rcpp::Nullable<arma::vec> pi_stars = R_NilValue,
                     const Rcpp::Nullable<arma::vec> Svec = R_NilValue,
                     const Rcpp::Nullable<arma::vec> Gvec = R_NilValue
                     ){

  arma::cube Y;
  if(model == "DINA"){
    if(itempars.isNull()){Rcpp::stop("Error: argument 'itempars' is missing");}
    Y = simDINA_g(alphas,Rcpp::as<arma::mat>(itempars),Q_matrix,Design_array);
  }
  if(model == "rRUM"){
    if(r_stars.isNull()){Rcpp::stop("Error: argument 'r_stars' is missing");}
    if(pi_stars.isNull()){Rcpp::stop("Error: argument 'pi_stars' is missing");}
    Y = simrRUM_g(alphas,Rcpp::as<arma::mat>(r_stars),Rcpp::as<arma::vec>(pi_stars),Q_matrix,Design_array);
  }
  if(model == "NIDA"){
    if(Svec.isNull()){Rcpp::stop("Error: argument 'Svec' is missing");}
    if(Gvec.isNull()){Rcpp::stop("Error: argument 'Gvec' is missing");}
    Y = simNIDA_g(alphas, Rcpp::as<arma::vec>(Svec), Rcpp::as<arma::vec>(Gvec),Q_matrix,Design_array);
  }

  return(Y);
}
