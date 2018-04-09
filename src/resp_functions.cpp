#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "resp_functions.h"

// -------------------------------- Response Model Functions -----------------------------------------------
// Simulating responses (single person and response cube),  computing likelihood of response vector                                          
// ---------------------------------------------------------------------------------------------------------


//' @title Simulate DINA model responses (single vector)
//' @description Simulate a single vector of DINA responses for a person on a set of items
//' @param J An \code{int} of number of items
//' @param K An \code{int} of number of attributes
//' @param ETA A \code{matrix} of ideal responses generated with ETAmat function
//' @param Svec A length J \code{vector} of item slipping parameters
//' @param Gvec A length J \code{vector} of item guessing parameters
//' @param alpha A length K \code{vector} of attribute pattern of a person 
//' @return A length J \code{vector} of item responses 
//' @examples
//' J = 15
//' K = 4
//' Q = random_Q(J,K)
//' ETA = ETAmat(K,J,Q)
//' s = runif(J,.1,.2)
//' g = runif(J,.1,.2)
//' alpha_i = c(1,0,0,1)
//' Y_i = sim_resp_DINA(J,K,ETA,s,g,alpha_i)
//' @export
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


//' @title Simulate DINA model responses (entire cube)
//' @description Simulate a cube of DINA responses for all persons on items across all time points
//' @param alphas An N-by-K-by-T \code{array} of attribute patterns of all persons across T time points 
//' @param itempars A J-by-2-by-T \code{cube} of item parameters (slipping: 1st col, guessin: 2nd col) across item blocks
//' @param ETA A J-by-2^K-by-T \code{array} of ideal responses across all item blocks, with each slice generated with ETAmat function
//' @param test_order A N_versions-by-T \code{matrix} indicating which block of items were administered to examinees with specific test version.
//' @param Test_versions A length N \code{vector} of the test version of each examinee
//' @return An \code{array} of DINA item responses of examinees across all time points
//' @examples
//' itempars_true <- array(runif(Jt*2*T,.1,.2), dim = c(Jt,2,T))
//' 
//' ETAs <- array(NA,dim = c(Jt,2^K,T)) 
//' for(t in 1:T){
//'   ETAs[,,t] <- ETAmat(K,Jt,Q_list[[t]])
//' }
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
//' Y_sim <- simDINA(Alphas,itempars_true,ETAs,test_order,Test_versions)
//' @export
// [[Rcpp::export]]
arma::cube simDINA(const arma::cube& alphas, const arma::cube& itempars, const arma::cube& ETA,
                   const arma::mat& test_order, const arma::vec& Test_versions){
  unsigned int N = alphas.n_rows;
  unsigned int Jt = itempars.n_rows;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::cube Y(N,Jt,T);
  arma::vec svec,gvec;
  arma::vec vv = bijectionvector(K);
  for(unsigned int i=0;i<N;i++){
    int test_version_i = Test_versions(i)-1;
    for(unsigned int t=0;t<T;t++){
      int test_block_it = test_order(test_version_i,t)-1;
      svec = itempars.slice(test_block_it).col(0);
      gvec = itempars.slice(test_block_it).col(1);
      arma::vec one_m_s = arma::ones<arma::vec>(Jt) - svec;
      double class_it = arma::dot(alphas.slice(t).row(i),vv);
      arma::vec eta_it = ETA.slice(test_block_it).col(class_it);
      arma::vec us = arma::randu<arma::vec>(Jt);
      arma::vec one_m_eta = arma::ones<arma::vec>(Jt) - eta_it;
      arma::vec ps = one_m_s%eta_it + gvec%one_m_eta;
      arma::vec compare = arma::zeros<arma::vec>(Jt);
      compare.elem(arma::find(ps-us > 0) ).fill(1.0);
      Y.subcube(i,0,t,i,Jt-1,t) = compare;
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



//' @title Simulate rRUM model responses (single vector)
//' @description Simulate a single vector of rRUM responses for a person on a set of items
//' @param J An \code{int} of number of items
//' @param K An \code{int} of number of attributes
//' @param Q A J-by-K Q \code{matrix}
//' @param rstar A J-by-K \code{matrix} of item penalty parameters for missing requisite skills
//' @param pistar length J \code{vector} of item correct response probability with all requisite skills
//' @param alpha A length K \code{vector} of attribute pattern of a person 
//' @return A length J \code{vector} of item responses
//' @examples
//' J = 15
//' K = 4
//' Q = random_Q(J,K)
//' Smats <- matrix(runif(J*K,.1,.3),J,K)
//' Gmats <- matrix(runif(J*K,.1,.3),J,K)
//' r_stars <- matrix(NA,J,K)
//' pi_stars <- numeric(J)
//' for(t in 1:T){
//'   pi_stars <- apply(((1-Smats)^Q),1,prod)
//'   r_stars <- Gmats/(1-Smats)
//' }
//' alpha_i = c(1,0,0,1)
//' Y_i = sim_resp_rRUM(J,K,Q,r_stars,pi_stars,alpha_i)
//' @export
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

//' @title Simulate rRUM model responses (entire cube)
//' @description Simulate a cube of rRUM responses for all persons on items across all time points
//' @param alphas An N-by-K-by-T \code{array} of attribute patterns of all persons across T time points 
//' @param r_stars A J-by-K-by-T \code{cube} of item penalty parameters for missing skills across all item blocks
//' @param pi_stars A J-by-T \code{matrix} of item correct response probability with all requisite skills across blocks
//' @param Qs A J-by-K-by-T  \code{cube} of Q-matrices across all item blocks
//' @param test_order A N_versions-by-T \code{matrix} indicating which block of items were administered to examinees with specific test version.
//' @param Test_versions A length N \code{vector} of the test version of each examinee
//' @return An \code{array} of rRUM item responses of examinees across all time points
//' @examples
//' Smats <- array(runif(Jt*K*(T),.1,.3),c(Jt,K,(T)))
//' Gmats <- array(runif(Jt*K*(T),.1,.3),c(Jt,K,(T)))
//' r_stars <- array(NA,c(Jt,K,T))
//' pi_stars <- matrix(NA,Jt,(T))
//' for(t in 1:T){
//'   pi_stars[,t] <- apply(((1-Smats[,,t])^Qs[,,t]),1,prod)
//'   r_stars[,,t] <- Gmats[,,t]/(1-Smats[,,t])
//' }
//' Test_versions_sim <- sample(1:5,N,replace = T)
//' tau <- numeric(K)
//'   for(k in 1:K){
//'     tau[k] <- runif(1,.2,.6)
//'   }
//'   R = matrix(0,K,K)
//' # Initial alphas
//' p_mastery <- c(.5,.5,.4,.4)
//' Alphas_0 <- matrix(0,N,K)
//' for(i in 1:N){
//'   for(k in 1:K){
//'     prereqs <- which(R[k,]==1)
//'     if(length(prereqs)==0){
//'       Alphas_0[i,k] <- rbinom(1,1,p_mastery[k])
//'     }
//'     if(length(prereqs)>0){
//'       Alphas_0[i,k] <- prod(Alphas_0[i,prereqs])*rbinom(1,1,p_mastery)
//'     }
//'   }
//' }
//' Alphas <- simulate_alphas_indept(tau,Alphas_0,T,R) 
//' Y_sim = simrRUM(Alphas,r_stars,pi_stars,Qs,test_order,Test_versions_sim)
//' @export
// [[Rcpp::export]]
arma::cube simrRUM(const arma::cube& alphas, const arma::cube& r_stars, const arma::mat& pi_stars, 
                   const arma::cube Qs, const arma::mat& test_order, const arma::vec& Test_versions){
  unsigned int N = alphas.n_rows;
  unsigned int Jt = pi_stars.n_rows;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::cube Y(N,Jt,T);
  for(unsigned int i=0;i<N;i++){
    int test_version_i = Test_versions(i)-1;
    for(unsigned int t=0;t<T;t++){
      int test_block_it = test_order(test_version_i,t)-1;
      arma::mat Q_it = Qs.slice(test_block_it);
      arma::mat rstar_it = r_stars.slice(test_block_it);
      arma::vec pistar_it = pi_stars.col(test_block_it);
      arma::vec alpha_it = alphas.slice(t).row(i).t();
      Y.slice(t).row(i) = sim_resp_rRUM(Jt,K,Q_it,rstar_it,pistar_it,alpha_it).t();
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

//' @title Simulate NIDA model responses (single vector)
//' @description Simulate a single vector of NIDA responses for a person on a set of items
//' @param J An \code{int} of number of items
//' @param K An \code{int} of number of attributes
//' @param Q A J-by-K Q \code{matrix}
//' @param Svec A length K \code{vector} of slipping probability in applying mastered skills
//' @param Gvec A length K \code{vector} of guessing probability in applying mastered skills
//' @param alpha A length K \code{vector} of attribute pattern of a person 
//' @return A length J \code{vector} of item responses
//' @examples
//' J = 15
//' K = 4
//' Q = random_Q(J,K)
//' Svec <- runif(K,.1,.3)
//' Gvec <- runif(K,.1,.3)
//' alpha_i = c(1,0,0,1)
//' Y_i = sim_resp_NIDA(J,K,Q,Svec,Gvec,alpha_i)
//' @export
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

//' @title Simulate NIDA model responses (entire cube)
//' @description Simulate a cube of NIDA responses for all persons on items across all time points
//' @param alphas An N-by-K-by-T \code{array} of attribute patterns of all persons across T time points 
//' @param Svec A length K \code{vector} of slipping probability in applying mastered skills
//' @param Gvec A length K \code{vector} of guessing probability in applying mastered skills
//' @param Qs A J-by-K-by-T  \code{cube} of Q-matrices across all item blocks
//' @param test_order A N_versions-by-T \code{matrix} indicating which block of items were administered to examinees with specific test version.
//' @param Test_versions A length N \code{vector} of the test version of each examinee
//' @return An \code{array} of NIDA item responses of examinees across all time points
//' @examples
//' Svec <- runif(K,.1,.3)
//' Gvec <- runif(K,.1,.3)
//' Test_versions_sim <- sample(1:5,N,replace = T)
//' tau <- numeric(K)
//'   for(k in 1:K){
//'     tau[k] <- runif(1,.2,.6)
//'   }
//'   R = matrix(0,K,K)
//' # Initial alphas
//'     p_mastery <- c(.5,.5,.4,.4)
//'     Alphas_0 <- matrix(0,N,K)
//'     for(i in 1:N){
//'       for(k in 1:K){
//'         prereqs <- which(R[k,]==1)
//'         if(length(prereqs)==0){
//'           Alphas_0[i,k] <- rbinom(1,1,p_mastery[k])
//'         }
//'         if(length(prereqs)>0){
//'           Alphas_0[i,k] <- prod(Alphas_0[i,prereqs])*rbinom(1,1,p_mastery)
//'         }
//'       }
//'     }
//'    Alphas <- simulate_alphas_indept(tau,Alphas_0,T,R) 
//' Y_sim = simNIDA(Alphas,Svec,Gvec,Qs,test_order,Test_versions_sim)
//' @export
// [[Rcpp::export]]
arma::cube simNIDA(const arma::cube& alphas, const arma::vec& Svec, const arma::vec& Gvec, 
                   const arma::cube Qs, const arma::mat& test_order, const arma::vec& Test_versions){
  unsigned int N = alphas.n_rows;
  unsigned int Jt = Qs.n_rows;
  unsigned int K = alphas.n_cols;
  unsigned int T = alphas.n_slices;
  arma::cube Y(N,Jt,T);
  for(unsigned int i=0;i<N;i++){
    int test_version_i = Test_versions(i)-1;
    for(unsigned int t=0;t<T;t++){
      int test_block_it = test_order(test_version_i,t)-1;
      arma::mat Q_it = Qs.slice(test_block_it);
      arma::vec alpha_it = alphas.slice(t).row(i).t();
      Y.slice(t).row(i) = sim_resp_NIDA(Jt,K,Q_it,Svec,Gvec,alpha_it).t();
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
