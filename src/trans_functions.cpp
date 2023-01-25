#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "trans_functions.h"
#include <RcppArmadilloExtensions/sample.h>

// -------------------------------- Transition Model Functions ----------------------------------------------
// Simulating transitions, computing transition probabilities                                                
// ----------------------------------------------------------------------------------------------------------


// [[Rcpp::export]]
arma::cube simulate_alphas_HO_sep_g(const arma::vec& lambdas, const arma::vec& thetas,
                                    const arma::mat& Q_matrix, const arma::cube& Design_array, 
                                    const arma::mat alpha0) {
  unsigned int K = Q_matrix.n_cols;
  unsigned int N = Design_array.n_rows;
  unsigned int L = Design_array.n_slices;
  arma::mat Jt(N,L);
  for(unsigned int i = 0; i<N; i++){
    for(unsigned int t = 0; t<L; t++){
      double Jt_it = arma::sum(Design_array.slice(t).row(i) == 1);
      Jt(i,t) = Jt_it;
    }
  }
  arma::mat alpha0s = alpha0;

  arma::vec alpha_i_prev;
  double theta_i;
  double sum_alpha_i;               // # mastered skills other than skill k
  double practice;                  // amount of practice on skill k
  double ex;
  double prob;
  unsigned int k;
  arma::vec alpha_i_new(K);
  Rcpp::List Q_examinee = Q_list_g(Q_matrix, Design_array);
  arma::cube alphas_all(N,K,L);
  alphas_all.slice(0) = alpha0s;
  for(unsigned int i = 0; i<N; i++){
    arma::mat Q_i = Q_examinee[i];
    arma::vec Jt_i = cumsum(Jt.row(i).t());
    for(unsigned int t = 1; t<L; t++){
      alpha_i_prev = alpha_i_new = alphas_all.slice(t-1).row(i).t();
      arma::uvec nonmastery = arma::find(alpha_i_prev == 0);
      if(nonmastery.n_elem>0){
        for(unsigned int kk = 0; kk<nonmastery.n_elem; kk++){
          k = nonmastery(kk);
          theta_i = thetas(i);
          sum_alpha_i = arma::sum(alpha_i_prev);
          arma::mat subQ = Q_i.rows(0, (Jt_i(t-1) - 1));
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
double pTran_HO_sep_g(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                    const arma::mat& Q_i, const arma::cube& Design_array, unsigned int t, unsigned int i) {
  unsigned int K = alpha_prev.n_elem;
  unsigned int L = Design_array.n_slices;

  arma::vec Jt_vec(L);
  for(unsigned int l=0; l<L; l++){
    Jt_vec(l) = arma::sum(Design_array.slice(l).row(i) == 1);
  }
  arma::vec Jt_i = cumsum(Jt_vec);
  
  arma::vec ptrans(K);
  double sum_alpha_i, ex;
  double practice = 0;
  for (unsigned int k = 0; k <K; k++) {
    if (alpha_prev(k) == 1) {
      ptrans(k) = 1;
    }
    if (alpha_prev(k) == 0) {
      sum_alpha_i = arma::sum(alpha_prev);
      
      practice = sum(Q_i.rows(0, (Jt_i(t) - 1)).col(k));
      
      ex = exp(lambdas(0) + lambdas(1)*theta_i + lambdas(2)*sum_alpha_i + lambdas(3)*practice);
      ptrans(k) = ex / (1 + ex);
    }
  }
  arma::vec prob = ptrans%alpha_post + (1 - ptrans) % (1 - alpha_post);
  return(arma::prod(prob));
}




// [[Rcpp::export]]
arma::cube simulate_alphas_HO_joint_g(const arma::vec& lambdas, const arma::vec& thetas,
                                      const arma::mat& Q_matrix, const arma::cube& Design_array, 
                                      const arma::mat& alpha0){
  unsigned int K = Q_matrix.n_cols;
  unsigned int N = Design_array.n_rows;
  unsigned int L = Design_array.n_slices;
  arma::mat Jt(N,L);
  for(unsigned int i = 0; i<N; i++){
    for(unsigned int t = 0; t<L; t++){
      double Jt_it = arma::sum(Design_array.slice(t).row(i) == 1);
      Jt(i,t) = Jt_it;
    }
  }
  arma::mat alpha0s = alpha0;

  Rcpp::List Q_examinee = Q_list_g(Q_matrix, Design_array);
  arma::cube alphas_all(N,K,L);
  alphas_all.slice(0) = alpha0s;
  arma::vec alpha_i_prev;
  double theta_i;
  double sum_alpha_i;               // # mastered skills other than skill k
  double practice;                  // amount of practice on skill k
  double ex;
  double prob;
  unsigned int k;
  arma::vec alpha_i_new(K);
  arma::vec Jt_i;
  arma::uvec nonmastery;
  arma::mat subQ;
  
  for(unsigned int i = 0; i<N; i++){
    arma::mat Q_i = Q_examinee[i];
    Jt_i = cumsum(Jt.row(i).t());
    for(unsigned int t = 1; t<L; t++){
      alpha_i_prev = alpha_i_new = alphas_all.slice(t-1).row(i).t();
      nonmastery = arma::find(alpha_i_prev == 0);
      if(nonmastery.n_elem>0){
        for(unsigned int kk = 0; kk<nonmastery.n_elem; kk++){
          k = nonmastery(kk);
          theta_i = thetas(i);
          sum_alpha_i = arma::sum(alpha_i_prev);
          subQ = Q_i.rows(0, (Jt_i(t-1) - 1));
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
double pTran_HO_joint_g(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                      const arma::mat& Q_i, const arma::cube Design_array, unsigned int t, unsigned int i){
  unsigned int K = alpha_prev.n_elem;
  unsigned int L = Design_array.n_slices;

  arma::vec Jt_vec(L);
  for(unsigned int t=0; t<L; t++){
    Jt_vec(t) = arma::sum(Design_array.slice(t).row(i) == 1);
  }
  arma::vec Jt_i = cumsum(Jt_vec);
  
  arma::vec ptrans(K);
  double sum_alpha_i, ex;
  double practice = 0;
  for(unsigned int k = 0; k <K; k++){
    if(alpha_prev(k)==1){
      ptrans(k)=1;
    }
    if(alpha_prev(k)==0){
      sum_alpha_i = arma::sum(alpha_prev);
      
      practice = sum(Q_i.rows(0, (Jt_i(t) - 1)).col(k));

      ex = exp(lambdas(0) + theta_i + lambdas(1)*sum_alpha_i + lambdas(2)*practice);
      ptrans(k) = ex/(1+ex);
    }
  }
  arma::vec prob = ptrans%alpha_post + (1-ptrans)%(1-alpha_post);
  return(arma::prod(prob));
}

// [[Rcpp::export]]
arma::cube simulate_alphas_indept(const arma::vec taus, const arma::mat& alpha0s, const unsigned int L, const arma::mat& R){
  unsigned int K = alpha0s.n_cols;
  unsigned int N = alpha0s.n_rows;
  arma::cube alphas_all(N,K,L);
  alphas_all.slice(0) = alpha0s;
  arma::vec alpha_i_prev;
  
  for(unsigned int i = 0; i<N; i++){
    for(unsigned int t = 1; t<L; t++){
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

// [[Rcpp::export]]
arma::cube simulate_alphas_indept_g(const arma::vec taus, const unsigned int N, const unsigned int L, const arma::mat& R,
                                    arma::mat alpha0){
  unsigned int K = R.n_cols;
  arma::mat alpha0s = alpha0;

  arma::cube alphas_all(N,K,L);
  alphas_all.slice(0) = alpha0s;
  arma::vec alpha_i_prev;
  
  for(unsigned int i = 0; i<N; i++){
    for(unsigned int t = 1; t<L; t++){
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


// [[Rcpp::export]]
arma::cube simulate_alphas_FOHM(const arma::mat& Omega,unsigned int N,unsigned int L,
                                const arma::mat alpha0){
  //unsigned int C = Omega.n_cols;
  //double u = R::runif(0,1);
  unsigned int K = log2(Omega.n_rows);
  arma::mat alpha0s = alpha0;
  arma::mat Alpha(N,L);
  arma::cube Alphas(N,K,L);
  Alphas.slice(0) = alpha0s;
  arma::vec alpha1 = alpha0s * bijectionvector(K);
  Alpha.col(0) = alpha1;
  
  for(unsigned int t=0;t<(L-1);t++){
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
arma::mat rAlpha(const arma::mat& Omega,unsigned int N,unsigned int L,
                 const arma::vec& alpha1){
  //unsigned int C = Omega.n_cols;
  //double u = R::runif(0,1);
  arma::mat Alpha(N,L);
  Alpha.col(0) = alpha1;
  
  for(unsigned int t=0;t<L-1;t++){
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
//' @return A 2^K-by-2^K transition matrix, the (i,j)th element indicating the transition probability of transitioning from i-th class to j-th class.
//' @examples
//' K = ncol(Q_matrix)
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


//' @title Generate attribute trajectories under the specified hidden Markov models
//' @description Based on the learning model parameters, create cube of attribute patterns
//' of all subjects across time. 
//' Currently available learning models are Higher-order hidden Markov DCM('HO_sep'), 
//' Higher-order hidden Markov DCM with learning ability as a random effect('HO_joint'), 
//' the simple independent-attribute learning model('indept'), 
//' and the first order hidden Markov model('FOHM'). 
//' @param model The learning model under which the attribute trajectories are generated. Available options are: 'HO_joint', 'HO_sep', 'indept', 'FOHM'.
//' @param lambdas A \code{vector} of transition model coefficients. With 'HO_sep' model specification, `lambdas` should be a length 4 \code{vector}. First entry is intercept of the logistic transition
//' model, second entry is the slope of general learning ability, third entry is the slope for number of other mastered skills,
//' fourth entry is the slope for amount of practice.
//' With 'HO_joint' model specification, `lambdas` should be a length 3 \code{vector}. First entry is intercept of the logistic transition
//' model, second entry is the slope for number of other mastered skills, third entry is the slope for amount of practice.
//' @param thetas A length N \code{vector} of learning abilities of each subject.
//' @param Q_matrix A J-by-K Q-matrix
//' @param Design_array A N-by-J-by-L array indicating items administered to examinee n at time point l.
//' @param taus A length K \code{vector} of transition probabilities from 0 to 1 on each skill
//' @param Omega A 2^K-by-2^K \code{matrix} of transition probabilities from row pattern to column pattern
//' @param N An \code{int} of number of examinees.
//' @param L An \code{int} of number of time points.
//' @param R A K-by-K dichotomous reachability \code{matrix} indicating the attribute hierarchies. The k,k'th entry of R is 1 if k' is prereq to k.
//' @param alpha0 Optional. An N-by-K \code{matrix} of subjects' initial attribute patterns.
//' @return An N-by-K-by-L \code{array} of attribute patterns of subjects at each time point.
//' @examples
//' \donttest{
//' ## HO_joint ##
//' N = nrow(Design_array)
//' J = nrow(Q_matrix)
//' K = ncol(Q_matrix)
//' L = dim(Design_array)[3]
//' class_0 <- sample(1:2^K, N, replace = TRUE)
//' Alphas_0 <- matrix(0,N,K)
//' for(i in 1:N){
//'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
//' }
//' thetas_true = rnorm(N, 0, 1.8)
//' lambdas_true <- c(-2, .4, .055)
//' Alphas <- sim_alphas(model="HO_joint", 
//'                     lambdas=lambdas_true, 
//'                     thetas=thetas_true, 
//'                     Q_matrix=Q_matrix, 
//'                     Design_array=Design_array)
//' 
//' ## HO_sep ##
//' N = dim(Design_array)[1]
//' J = nrow(Q_matrix)
//' K = ncol(Q_matrix)
//' L = dim(Design_array)[3]
//' class_0 <- sample(1:2^K, N, replace = L)
//' Alphas_0 <- matrix(0,N,K)
//' for(i in 1:N){
//'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
//' }
//' thetas_true = rnorm(N)
//' lambdas_true = c(-1, 1.8, .277, .055)
//' Alphas <- sim_alphas(model="HO_sep", 
//'                      lambdas=lambdas_true, 
//'                      thetas=thetas_true, 
//'                      Q_matrix=Q_matrix, 
//'                      Design_array=Design_array)
//' 
//' ## indept ##
//' N = dim(Design_array)[1]
//' K = dim(Q_matrix)[2]
//' L = dim(Design_array)[3]
//' tau <- numeric(K)
//' for(k in 1:K){
//'   tau[k] <- runif(1,.2,.6)
//' }
//' R = matrix(0,K,K)
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
//' Alphas <- sim_alphas(model="indept", taus=tau, N=N, L=L, R=R)
//' 
//' ## FOHM ##
//' N = dim(Design_array)[1]
//' K = ncol(Q_matrix)
//' L = dim(Design_array)[3]
//' TP <- TPmat(K)
//' Omega_true <- rOmega(TP)
//' class_0 <- sample(1:2^K, N, replace = L)
//' Alphas_0 <- matrix(0,N,K)
//' for(i in 1:N){
//'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
//' }
//' Alphas <- sim_alphas(model="FOHM", Omega = Omega_true, N=N, L=L)
//' }
//' @export
// [[Rcpp::export]]
arma::cube sim_alphas(const std::string model,
                     const Rcpp::Nullable<arma::vec&> lambdas = R_NilValue, const Rcpp::Nullable<arma::vec&> thetas = R_NilValue,
                     const Rcpp::Nullable<arma::mat&> Q_matrix = R_NilValue, const Rcpp::Nullable<arma::cube&> Design_array = R_NilValue, 
                     const Rcpp::Nullable<arma::vec> taus = R_NilValue, const Rcpp::Nullable<arma::mat&> Omega = R_NilValue,
                     int N = NA_INTEGER, const int L = NA_INTEGER, 
                     const Rcpp::Nullable<arma::mat&> R = R_NilValue,
                     const Rcpp::Nullable<arma::mat> alpha0 = R_NilValue){
  
  if(model == "HO_joint"){
    if(lambdas.isNull()){Rcpp::stop("Error: argument 'lambdas' is missing");}
    if(thetas.isNull()){Rcpp::stop("Error: argument 'thetas' is missing");}
    if(Q_matrix.isNull()){Rcpp::stop("Error: argument 'Q_matrix' is missing");}
    if(Design_array.isNull()){Rcpp::stop("Error: argument 'Design_array' is missing");}
  }
  if(model == "HO_sep"){
    if(lambdas.isNull()){Rcpp::stop("Error: argument 'lambdas' is missing");}
    if(thetas.isNull()){Rcpp::stop("Error: argument 'thetas' is missing");}
    if(Q_matrix.isNull()){Rcpp::stop("Error: argument 'Q_matrix' is missing");}
    if(Design_array.isNull()){Rcpp::stop("Error: argument 'Design_array' is missing");}
  }
  if(model == "indept"){
    if(taus.isNull()){Rcpp::stop("Error: argument 'taus' is missing");}
    if(N == NA_INTEGER){Rcpp::stop("Error: argument 'N' is missing");}
    if(L == NA_INTEGER){Rcpp::stop("Error: argument 'L' is missing");}
    if(R.isNull()){Rcpp::stop("Error: argument 'R' is missing");}
  }
  if(model == "FOHM"){
    if(Omega.isNull()){Rcpp::stop("Error: argument 'Omega' is missing");}
    if(N == NA_INTEGER){Rcpp::stop("Error: argument 'N' is missing");}
    if(L == NA_INTEGER){Rcpp::stop("Error: argument 'L' is missing");}
  }
  arma::vec lambdas_temp;
  arma::vec thetas_temp;
  arma::mat Q_matrix_temp;
  arma::cube Design_array_temp;
  arma::vec taus_temp;
  arma::mat Omega_temp;
  arma::mat R_temp;
  unsigned int K;
  
  if(lambdas.isNotNull()){lambdas_temp = Rcpp::as<arma::vec>(lambdas);}
  if(thetas.isNotNull()){thetas_temp = Rcpp::as<arma::vec>(thetas);}
  if(Q_matrix.isNotNull()){
    Q_matrix_temp = Rcpp::as<arma::mat>(Q_matrix);
    K = Q_matrix_temp.n_cols;
    }
  if(Design_array.isNotNull()){
    Design_array_temp = Rcpp::as<arma::cube>(Design_array);
    N = Design_array_temp.n_rows;
    }
  if(taus.isNotNull()){taus_temp = Rcpp::as<arma::vec>(taus);}
  if(Omega.isNotNull()){
    Omega_temp = Rcpp::as<arma::mat>(Omega);
    K = log2(Omega_temp.n_rows);
    }
  if(R.isNotNull()){
    R_temp = Rcpp::as<arma::mat>(R);
    K = R_temp.n_cols;
    }
  
  arma::mat alpha0s(N,K);
  if(alpha0.isNull()){
    arma::uvec class_0 = Rcpp::RcppArmadillo::sample(arma::linspace<arma::uvec>(1, 2^K), N, true);
    for(unsigned int i=0; i<N; i++){
      alpha0s.row(i) = inv_bijectionvector(K, class_0[i]-1).t();
    }
  }else{
    alpha0s = Rcpp::as<arma::mat>(alpha0);
  }
  
  arma::cube Alphas;
  if(model == "HO_joint"){
    Alphas = simulate_alphas_HO_joint_g(lambdas_temp, thetas_temp, Q_matrix_temp, Design_array_temp, alpha0s);
  }

  if(model == "HO_sep"){
    Alphas = simulate_alphas_HO_sep_g(lambdas_temp, thetas_temp, Q_matrix_temp, Design_array_temp, alpha0s);
  }

  if(model == "indept"){
    Alphas = simulate_alphas_indept_g(taus_temp, N, L, R_temp, alpha0s);
  }

  if(model == "FOHM"){
    Alphas = simulate_alphas_FOHM(Omega_temp, N, L, alpha0s);
  }
  
  return Alphas;
} 
