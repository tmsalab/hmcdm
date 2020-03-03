#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "trans_functions.h"


// -------------------------------- Transition Model Functions ----------------------------------------------
// Simulating transitions, computing transition probabilities                                                
// ----------------------------------------------------------------------------------------------------------

//' @title Generate attribute trajectories under the Higher-Order Hidden Markov DCM
//' @description Based on the initial attribute patterns and learning model parameters, create cube of attribute patterns
//' of all subjects across time. General learning ability is regarded as a fixed effect and has a slope.
//' @param lambdas A length 4 \code{vector} of transition model coefficients. First entry is intercept of the logistic transition
//' model, second entry is the slope of general learning ability, third entry is the slope for number of other mastered skills,
//' fourth entry is the slope for amount of practice.
//' @param thetas A length N \code{vector} of learning abilities of each subject.
//' @param alpha0s An N-by-K \code{matrix} of subjects' initial attribute patterns.
//' @param Q_examinee A length N \code{list} of Jt*K Q matrices across time for each examinee, items are in the order that they are
//' administered to the examinee
//' @param T An \code{int} of number of time points
//' @param Jt An \code{int} of number of items in each block
//' @return An N-by-K-by-T \code{array} of attribute patterns of subjects at each time point.
//' @examples
//' N = length(Test_versions)
//' Jt = nrow(Q_list[[1]])
//' K = ncol(Q_list[[1]])
//' T = nrow(test_order)
//' J = Jt*T
//' class_0 <- sample(1:2^K, N, replace = T)
//' Alphas_0 <- matrix(0,N,K)
//' thetas_true = rnorm(N)
//' for(i in 1:N){
//'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
//' }
//' lambdas_true = c(-1, 1.8, .277, .055)
//' Alphas <- simulate_alphas_HO_sep(lambdas_true,thetas_true,Alphas_0,Q_examinee,T,Jt)
//' @export
// [[Rcpp::export]]
arma::cube simulate_alphas_HO_sep(const arma::vec& lambdas, const arma::vec& thetas, const arma::mat& alpha0s,
                                  const Rcpp::List& Q_examinee, const unsigned int T, const unsigned int Jt){
  unsigned int K = alpha0s.n_cols;
  unsigned int N = alpha0s.n_rows;
  arma::cube alphas_all(N,K,T);
  alphas_all.slice(0) = alpha0s;
  arma::vec alpha_i_prev;
  double theta_i;
  double sum_alpha_i;               // # mastered skills other than skill k
  double practice;                  // amount of practice on skill k
  double ex;
  double prob;
  unsigned int k;
  arma::vec alpha_i_new(K);
  
  for(unsigned int i = 0; i<N; i++){
    arma::mat Q_i = Q_examinee[i];
    for(unsigned int t = 1; t<T; t++){
      alpha_i_prev = alpha_i_new = alphas_all.slice(t-1).row(i).t();
      arma::uvec nonmastery = arma::find(alpha_i_prev == 0);
      if(nonmastery.n_elem>0){
        for(unsigned int kk = 0; kk<nonmastery.n_elem; kk++){
          k = nonmastery(kk);
          theta_i = thetas(i);
          sum_alpha_i = arma::sum(alpha_i_prev);
          arma::mat subQ = Q_i.rows(0, (t*Jt - 1));
          practice = sum(subQ.col(kk));
          ex = exp(lambdas(0) + lambdas(1)*theta_i + lambdas(2)*sum_alpha_i + lambdas(3)*practice);
          prob = ex/(1+ex);
          double u = R::runif(0,1);
          if(u<prob){
            alpha_i_new(k) = 1;
          }
        }
      }
      alphas_all.slice(t).row(i) = alpha_i_new.t();
    }
  }
  return(alphas_all);
}


// [[Rcpp::export]]
double pTran_HO_sep(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                    const arma::mat& Q_i, unsigned int Jt, unsigned int t) {
  unsigned int K = alpha_prev.n_elem;
  arma::vec ptrans(K);
  double sum_alpha_i, ex;
  double practice = 0;
  for (unsigned int k = 0; k <K; k++) {
    if (alpha_prev(k) == 1) {
      ptrans(k) = 1;
    }
    if (alpha_prev(k) == 0) {
      sum_alpha_i = arma::sum(alpha_prev);
      
      arma::mat subQ = Q_i.rows(0, ((t+1)*Jt - 1));
      practice = sum(subQ.col(k));
      
      ex = exp(lambdas(0) + lambdas(1)*theta_i + lambdas(2)*sum_alpha_i + lambdas(3)*practice);
      ptrans(k) = ex / (1 + ex);
    }
  }
  arma::vec prob = ptrans%alpha_post + (1 - ptrans) % (1 - alpha_post);
  return(arma::prod(prob));
}




//' @title Generate attribute trajectories under the Higher-Order Hidden Markov DCM with latent learning ability as a random effect
//' @description Based on the initial attribute patterns and learning model parameters, create cube of attribute patterns
//' of all subjects across time. General learning ability is regarded as a random intercept.
//' @param lambdas A length 3 \code{vector} of transition model coefficients. First entry is intercept of the logistic transition
//' model, second entry is the slope for number of other mastered skills, third entry is the slope for amount of practice.
//' @param thetas A length N \code{vector} of learning abilities of each subject.
//' @param alpha0s An N-by-K \code{matrix} of subjects' initial attribute patterns.
//' @param Q_examinee A length N \code{list} of Jt*K Q matrices across time for each examinee, items are in the order that they are
//' administered to the examinee
//' @param T An \code{int} of number of time points
//' @param Jt An \code{int} of number of items in each block
//' @return An N-by-K-by-T \code{array} of attribute patterns of subjects at each time point.
//' @examples
//' N = length(Test_versions)
//' Jt = nrow(Q_list[[1]])
//' K = ncol(Q_list[[1]])
//' T = nrow(test_order)
//' J = Jt*T
//' class_0 <- sample(1:2^K, N, replace = T)
//' Alphas_0 <- matrix(0,N,K)
//' mu_thetatau = c(0,0)
//' Sig_thetatau = rbind(c(1.8^2,.4*.5*1.8),c(.4*.5*1.8,.25))
//' Z = matrix(rnorm(N*2),N,2)
//' thetatau_true = Z%*%chol(Sig_thetatau)
//' thetas_true = thetatau_true[,1]
//' for(i in 1:N){
//'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
//' }
//' lambdas_true <- c(-2, .4, .055)     
//' Alphas <- simulate_alphas_HO_joint(lambdas_true,thetas_true,Alphas_0,Q_examinee,T,Jt)
//' @export
// [[Rcpp::export]]
arma::cube simulate_alphas_HO_joint(const arma::vec& lambdas, const arma::vec& thetas, const arma::mat& alpha0s,
                                    const Rcpp::List& Q_examinee, const unsigned int T, const unsigned int Jt){
  unsigned int K = alpha0s.n_cols;
  unsigned int N = alpha0s.n_rows;
  arma::cube alphas_all(N,K,T);
  alphas_all.slice(0) = alpha0s;
  arma::vec alpha_i_prev;
  double theta_i;
  double sum_alpha_i;               // # mastered skills other than skill k
  double practice;                  // amount of practice on skill k
  double ex;
  double prob;
  unsigned int k;
  arma::vec alpha_i_new(K);
  
  for(unsigned int i = 0; i<N; i++){
    arma::mat Q_i = Q_examinee[i];
    for(unsigned int t = 1; t<T; t++){
      alpha_i_prev = alpha_i_new = alphas_all.slice(t-1).row(i).t();
      arma::uvec nonmastery = arma::find(alpha_i_prev == 0);
      if(nonmastery.n_elem>0){
        for(unsigned int kk = 0; kk<nonmastery.n_elem; kk++){
          k = nonmastery(kk);
          theta_i = thetas(i);
          sum_alpha_i = arma::sum(alpha_i_prev);
          arma::mat subQ = Q_i.rows(0, (t*Jt - 1));
          practice = sum(subQ.col(kk));
          ex = exp(lambdas(0) + theta_i + lambdas(1)*sum_alpha_i + lambdas(2)*practice);
          prob = ex/(1+ex);
          double u = R::runif(0,1);
          if(u<prob){
            alpha_i_new(k) = 1;
          }
        }
      }
      alphas_all.slice(t).row(i) = alpha_i_new.t();
    }
  }
  return(alphas_all);
}


// Transition probability of the HMDCM, with joint response and response time modeling (i.e., no slope for theta)
// [[Rcpp::export]]
double pTran_HO_joint(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                      const arma::mat& Q_i, unsigned int Jt, unsigned int t){
  unsigned int K = alpha_prev.n_elem;
  arma::vec ptrans(K);
  double sum_alpha_i, ex;
  double practice = 0;
  for(unsigned int k = 0; k <K; k++){
    if(alpha_prev(k)==1){
      ptrans(k)=1;
    }
    if(alpha_prev(k)==0){
      sum_alpha_i = arma::sum(alpha_prev);
      arma::mat subQ = Q_i.rows(0, ((t+1)*Jt - 1));
      practice = sum(subQ.col(k));
      
      ex = exp(lambdas(0) + theta_i + lambdas(1)*sum_alpha_i + lambdas(2)*practice);
      ptrans(k) = ex/(1+ex);
    }
  }
  arma::vec prob = ptrans%alpha_post + (1-ptrans)%(1-alpha_post);
  return(arma::prod(prob));
}






//' @title Generate attribute trajectories under the simple independent-attribute learning model
//' @description Based on the initial attribute patterns and probability of transitioning from 0 to 1 on each attribute, 
//' create cube of attribute patterns of all subjects across time. Transitions on different skills are regarded as independent.
//' @param taus A length K \code{vector} of transition probabilities from 0 to 1 on each skill
//' @param alpha0s An N-by-K \code{matrix} of subjects' initial attribute patterns.
//' @param T An \code{int} of number of time points
//' @param R A K-by-K dichotomous reachability \code{matrix} indicating the attribute hierarchies. The k,k'th entry of R is 1 if k' is prereq to k.
//' @return An N-by-K-by-T \code{array} of attribute patterns of subjects at each time point.
//' @examples
//' N = length(Test_versions)
//' Jt = nrow(Q_list[[1]])
//' K = ncol(Q_list[[1]])
//' T = nrow(test_order)
//' J = Jt*T
//' tau <- numeric(K)
//' for(k in 1:K){
//'   tau[k] <- runif(1,.2,.6)
//' }
//' R = matrix(0,K,K)
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
//' @export
// [[Rcpp::export]]
arma::cube simulate_alphas_indept(const arma::vec taus, const arma::mat& alpha0s, const unsigned int T, const arma::mat& R){
  unsigned int K = alpha0s.n_cols;
  unsigned int N = alpha0s.n_rows;
  arma::cube alphas_all(N,K,T);
  alphas_all.slice(0) = alpha0s;
  arma::vec alpha_i_prev;
  
  for(unsigned int i = 0; i<N; i++){
    for(unsigned int t = 1; t<T; t++){
      alpha_i_prev = alphas_all.slice(t-1).row(i).t();
      for(unsigned int k = 0; k<K; k++){
        arma::uvec prereqs = arma::find(R.row(k)==1);
        double u = R::runif(0,1);
        if(u<taus[k]){
          if(prereqs.n_elem == 0){
            alpha_i_prev(k) = 1; 
          }
          if(prereqs.n_elem >0){
            if(alpha_i_prev(k) == 0){
              alpha_i_prev(k) = arma::prod(alpha_i_prev(prereqs));
            }
          }
        }
      }
      alphas_all.slice(t).row(i) = alpha_i_prev.t();
    }
  }
  return(alphas_all);
}


// Transition probability
//[[Rcpp::export]]
double pTran_indept(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& taus,const arma::mat& R){
  double p = 1;
  double a_prereq_post = 1;
  double a_prereq_prev = 1;
  unsigned int K = alpha_prev.n_elem;
  for(unsigned int k = 0; k<K; k++){
    arma::uvec prereq = arma::find(R.row(k)==1);
    if(prereq.n_elem>0){
      a_prereq_post = arma::prod(alpha_post(prereq)); // 1 if no prereq or prereqs are learned, 0 else
      a_prereq_prev = arma::prod(alpha_prev(prereq));
    }
    if(alpha_prev(k)==1){
      p = p * alpha_post(k) * a_prereq_prev * a_prereq_post;
    }
    if(alpha_prev(k)==0){
      p = p * (alpha_post(k)*taus(k)*a_prereq_post+(1-alpha_post(k))*(1-taus(k)));
    }
  }
  return(p);
}


//' @title Generate attribute trajectories under the first order hidden Markov model
//' @description Based on the initial attribute patterns and probability of transitioning between different patterns, 
//' create cube of attribute patterns of all subjects across time. 
//' @param Omega A 2^K-by-2^K \code{matrix} of transition probabilities from row pattern to column attern
//' @param alpha0s An N-by-K \code{matrix} of subjects' initial attribute patterns.
//' @param T An \code{int} of number of time points
//' @return An N-by-K-by-T \code{array} of attribute patterns of subjects at each time point. 
//' @examples
//' N = length(Test_versions)
//' Jt = nrow(Q_list[[1]])
//' K = ncol(Q_list[[1]])
//' T = nrow(test_order)
//' J = Jt*T
//' TP <- TPmat(K)
//' Omega_true <- rOmega(TP)
//' class_0 <- sample(1:2^K, N, replace = T)
//' Alphas_0 <- matrix(0,N,K)
//' for(i in 1:N){
//'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
//' }
//' Alphas <- simulate_alphas_FOHM(Omega_true, Alphas_0,T)
//' @export
// [[Rcpp::export]]
arma::cube simulate_alphas_FOHM(const arma::mat& Omega,const arma::mat& alpha0s,unsigned int T){
  //unsigned int C = Omega.n_cols;
  //double u = R::runif(0,1);
  unsigned int N = alpha0s.n_rows;
  unsigned int K = alpha0s.n_cols;
  arma::mat Alpha(N,T);
  arma::cube Alphas(N,K,T);
  Alphas.slice(0) = alpha0s;
  arma::vec alpha1 = alpha0s * bijectionvector(K);
  Alpha.col(0) = alpha1;
  
  for(unsigned int t=0;t<(T-1);t++){
    for(unsigned int i=0;i<N;i++){
      double cl = Alpha(i,t);
      arma::uvec trans_classes = find(Omega.row(cl) > 0.0);
      arma::rowvec OmegaRow = Omega.row(cl);
      arma::vec wcc = OmegaRow.cols(trans_classes).t();
      double rcl = rmultinomial(wcc);
      Alpha(i,t+1) = trans_classes(rcl);
      Alphas.slice(t+1).row(i) = inv_bijectionvector(K,Alpha(i,t+1)).t();
    }
  }
  //  Rcpp::Rcout << wcc << std::endl;
  //  Rcpp::Rcout << rcl << CL << std::endl;
  return Alphas;
}

// [[Rcpp::export]]
arma::mat rAlpha(const arma::mat& Omega,unsigned int N,unsigned int T,
                 const arma::vec& alpha1){
  //unsigned int C = Omega.n_cols;
  //double u = R::runif(0,1);
  arma::mat Alpha(N,T);
  Alpha.col(0) = alpha1;
  
  for(unsigned int t=0;t<T-1;t++){
    for(unsigned int i=0;i<N;i++){
      double cl = Alpha(i,t);
      arma::uvec trans_classes = find(Omega.row(cl) > 0.0);
      arma::rowvec OmegaRow = Omega.row(cl);
      arma::vec wcc = OmegaRow.cols(trans_classes).t();
      double rcl = rmultinomial(wcc);
      //Rcpp::Rcout << i<<t <<trans_classes(rcl)<< std::endl;
      Alpha(i,t+1) = trans_classes(rcl);
    }
  }
  //  Rcpp::Rcout << wcc << std::endl;
  //  Rcpp::Rcout << rcl << CL << std::endl;
  return Alpha;
}


//' @title Generate a random transition matrix for the first order hidden Markov model
//' @description Generate a random transition matrix under nondecreasing learning trajectory assumption
//' @param TP A 2^K-by-2^K dichotomous matrix of indicating possible transitions under the monotonicity assumption, created with
//' the TPmat function 
//' @examples
//' N = length(Test_versions)
//' Jt = nrow(Q_list[[1]])
//' K = ncol(Q_list[[1]])
//' T = nrow(test_order)
//' J = Jt*T
//' TP = TPmat(K)
//' Omega_sim = rOmega(TP)
//' @export
// [[Rcpp::export]]
arma::mat rOmega(const arma::mat& TP){
  unsigned int C = TP.n_cols;
  arma::mat Omega = arma::zeros<arma::mat>(C,C);
  Omega(C-1,C-1) = 1.;
  for(unsigned int cc=0;cc<C-1;cc++){
    arma::uvec tflag = find(TP.row(cc)==1.);
    arma::vec delta0 = arma::ones<arma::vec>(tflag.n_elem);
    arma::vec ws = rDirichlet(delta0);
    
    //          Rcpp::Rcout << ws <<tflag<< std::endl;
    for(unsigned int g=0;g<tflag.n_elem;g++){
      Omega(cc,tflag(g)) = ws(g);
    }
  }
  return Omega;
}

//' @title Generate a random transition matrix for the first order hidden Markov model
//' @description Generate a random transition matrix under the unrestricted learning trajectory assumption
//' @param TP A 2^K-by-2^K dichotomous matrix of indicating possible transitions
//' @examples
//' K = ncol(Q_list[[1]])
//' TP = TPmatFree(K)
//' Omega_sim = rOmega(TP)
//' @export
// [[Rcpp::export]]
arma::mat rOmegaFree(const arma::mat& TP){
  unsigned int C = TP.n_cols;
  arma::mat Omega = arma::zeros<arma::mat>(C,C);
  Omega(C-1,C-1) = 1.;
  for(unsigned int cc=0;cc<C;cc++){
    arma::uvec tflag = find(TP.row(cc)==1.);
    arma::vec delta0 = arma::ones<arma::vec>(tflag.n_elem);
    arma::vec ws = rDirichlet(delta0);
    
    //          Rcpp::Rcout << ws <<tflag<< std::endl;
    for(unsigned int g=0;g<tflag.n_elem;g++){
      Omega(cc,tflag(g)) = ws(g);
    }
  }
  return Omega;
}

/*** R
# TP = TPmat(2)
# TPFree=TPmatFree(2)
# rOmega(TP)
# rOmegaFree(TPFree)
# Omega = rOmegaFree(TP)
# rowSums(Omega)

*/