#include <RcppArmadillo.h>
#include <RProgress.h>

using namespace Rcpp;





// ------------------------------------------- Basic Functions ---------------------------------------------------
// Functions for generating random numbers or generic functions for CDM (e.g., Q matrix, ETA, bijection, etc.)
// ---------------------------------------------------------------------------------------------------------------



// [[Rcpp::export]]
arma::vec bijectionvector(unsigned int K) {
  arma::vec vv(K);
  for(unsigned int k=0;k<K;k++){
    vv(k) = pow(2,K-k-1);
  }
  return vv;
}

//' @export
// [[Rcpp::export]]
arma::vec inv_bijectionvector(unsigned int K,double CL){
  arma::vec alpha(K);
  for(unsigned int k=0;k<K;k++){
    double twopow = pow(2,K-k-1);
    alpha(k) = (twopow<=CL);
    CL = CL - twopow*alpha(k);
  }
  return alpha;
}

// [[Rcpp::export]]
arma::mat rwishart(unsigned int df, const arma::mat& S) {
  // Dimension of returned wishart
  unsigned int m = S.n_rows;
  
  // Z composition:
  // sqrt chisqs on diagonal
  // random normals below diagonal
  // misc above diagonal
  arma::mat Z(m,m);
  
  // Fill the diagonal
  for(unsigned int i = 0; i < m; i++) {
    Z(i,i) = sqrt(R::rchisq(df-i));
  }
  
  // Fill the lower matrix with random guesses
  for(unsigned int j = 0; j < m; j++) {  
    for(unsigned int i = j+1; i < m; i++) {    
      Z(i,j) = R::rnorm(0,1);
    }}
  
  // Lower triangle * chol decomp
  arma::mat C = arma::trimatl(Z).t() * arma::chol(S);
  
  // Return random wishart
  return C.t()*C;
}


//' @title Generate Random Inverse Wishart Distribution
//' @description Creates a random inverse wishart distribution when given degrees of freedom and a sigma matrix. 
//' @param df An \code{int} that represents the degrees of freedom.  (> 0)
//' @param Sig A \code{matrix} with dimensions m x m that provides Sigma, the covariance matrix. 
//' @return A \code{matrix} that is an inverse wishart distribution.
//' @author James J Balamuta
//' @examples 
//' #Call with the following data:
//' riwishart(3, diag(2))
//' @export
// [[Rcpp::export]]
arma::mat rinvwish(unsigned int df, const arma::mat& Sig) {
  return rwishart(df,Sig.i()).i();
}

// [[Rcpp::export]]
double rmultinomial(const arma::vec& ps){
  unsigned int C = ps.n_elem;
  double u = R::runif(0,1);
  arma::vec cps = cumsum(ps);
  arma::vec Ips = arma::zeros<arma::vec>(C);
  
  Ips.elem(arma::find(cps < u) ).fill(1.0);
  
  return sum(Ips);
}

// [[Rcpp::export]]
arma::vec rDirichlet(const arma::vec& deltas){
  unsigned int C = deltas.n_elem;
  arma::vec Xgamma(C);
  
  //generating gamma(deltac,1)
  for(unsigned int c=0;c<C;c++){
    Xgamma(c) = R::rgamma(deltas(c),1.0);
  }
  return Xgamma/sum(Xgamma);
}

const double log2pi = std::log(2.0 * M_PI);

// [[Rcpp::export]]
double dmvnrm(arma::vec x,  
              arma::vec mean,  
              arma::mat sigma, 
              bool logd = false) { 
  int xdim = x.n_elem;
  double out;
  arma::mat rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  double rootisum = arma::sum(log(rooti.diag()));
  double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  
  arma::vec z = rooti * ( x - mean) ;    
  out = constants - 0.5 * arma::sum(z%z) + rootisum;     
  
  if (logd == false) {
    out = exp(out);
  }
  return(out);
}


// Multivariate normal random generation
// [[Rcpp::export]]
arma::vec rmvnrm(arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::vec Y = arma::randn(ncols);
  return mu + (Y.t() * arma::chol(sigma)).t();
}


//' @title Generate random Q matrix
//' @description Creates a random Q matrix containing three identity matrices after row permutation
//' @param J An \code{int} that represents the number of items
//' @param K An \code{int} that represents the number of attributes/skills
//' @return A dichotomous \code{matrix} for Q.
//' @examples 
//' random_Q(15,4)
//' @export
// [[Rcpp::export]]
arma::mat random_Q(unsigned int J,unsigned int K) {
  unsigned int nClass = pow(2,K);
  arma::vec vv = bijectionvector(K);
  arma::vec Q_biject(J);
  Q_biject(arma::span(0,K-1)) = vv;
  Q_biject(arma::span(K,2*K-1)) = vv;
  Q_biject(arma::span(2*K,3*K-1)) = vv;
  arma::vec Jm3K = arma::randi<arma::vec>(J-3*K,arma::distr_param(1,nClass-1) ) ;
  Q_biject(arma::span(3*K,J-1)) = Jm3K;
  Q_biject = arma::shuffle(Q_biject);
  arma::mat Q(J,K);
  for(unsigned int j=0;j<J;j++){
    arma::vec qj = inv_bijectionvector(K,Q_biject(j));
    Q.row(j) = qj.t();
  }
  return Q;
}


//' @title Generate ideal response matrix
//' @description Based on the Q matrix and the latent attribute space, generate the ideal response matrix for each skill pattern
//' @param K An \code{int} of the number of attributes
//' @param J An \code{int} of the number of items
//' @param Q A J-by-K Q \code{matrix}
//' @return A J-by-2^K ideal response \code{matrix}
//' @examples 
//' Q = random_Q(15,4)
//' ETA = ETAmat(4,15,Q)
//' @export
// [[Rcpp::export]]
arma::mat ETAmat(unsigned int K,unsigned int J,const arma::mat& Q) {
  double nClass = pow(2,K);
  arma::mat ETA(J,nClass);
  for(unsigned int cc=0;cc<nClass;cc++){                //*
    arma::vec alpha_c = inv_bijectionvector(K,cc);
    for(unsigned int j=0;j<J;j++){                  //*
      arma::rowvec qj = Q.row(j);
      double compare = arma::conv_to<double>::from(qj*alpha_c - qj*qj.t());
      ETA(j,cc) = (compare>=0);
    }
  }
  return ETA;
}



//' @title Generate monotonicity matrix
//' @description Based on the latent attribute space, generate a matrix indicating whether it is possible to
//' transition from pattern cc to cc' under the monotonicity learning assumption.
//' @param K An \code{int} of the number of attribtues.
//' @return A 2^K-by-2^K dichotomous \code{matrix} of whether it is possible to transition between two patterns 
//' @examples
//' TP = TPmat(4)
//' @export
// [[Rcpp::export]]
arma::mat TPmat(unsigned int K){
  double nClass = pow(2,K);
  arma::mat TP = arma::eye<arma::mat>(nClass,nClass);
  for(unsigned int rr=0;rr<nClass-1;rr++){
    for(unsigned int cc=rr+1;cc<nClass;cc++){
      arma::vec alpha_r = inv_bijectionvector(K,rr);
      arma::vec alpha_c = inv_bijectionvector(K,cc);
      double temp = 1.0;
      for(unsigned int k=0;k<K;k++){
        temp = (alpha_r(k)<=alpha_c(k))*temp;
      }
      TP(rr,cc) = temp;
    }
  }
  return TP;
}

// [[Rcpp::export]]
arma::mat crosstab(const arma::vec& V1,const arma::vec& V2,const arma::mat& TP,
                   unsigned int nClass,unsigned int col_dim){
  //  arma::uvec freq = arma::hist(V2_temp,arma::linspace(0,nC-1,nC));
  arma::mat CTmat = arma::zeros<arma::mat>(nClass,col_dim);
  for(unsigned int rr=0;rr<nClass;rr++){
    arma::uvec V1_c = find(V1 == rr);
    arma::vec V2_temp = V2(V1_c);
    arma::uvec cells = find(TP.row(rr) == 1);
    
    for(unsigned int cc=0;cc<cells.n_elem;cc++){
      double ccc = cells(cc);
      arma::uvec V2_temp_c = find(V2_temp!=ccc);
      CTmat(rr,ccc) = V2_temp.n_elem - V2_temp_c.n_elem;
      V2_temp = V2_temp(V2_temp_c);
    }
  }
  
  return CTmat;
}


// [[Rcpp::export]]
arma::cube resp_miss(const arma::cube& Responses, const arma::mat& test_order, 
                     const arma::vec& Test_versions){
  unsigned int Jt = Responses.n_cols;
  unsigned int T = Responses.n_slices;
  unsigned int N = Responses.n_rows;
  unsigned int J = Jt*T;
  
  arma::cube Y_miss(N,J,T);
  Y_miss.fill(NA_REAL);
  
  for(unsigned int i = 0; i<N; i++){
    unsigned int Test_version_i = Test_versions(i)-1;
    for(unsigned int t = 0; t<T; t++){
      unsigned int test_block_it = test_order(Test_version_i,t)-1;
      Y_miss.subcube(i,(test_block_it*Jt),t,i,((test_block_it+1)*Jt-1),t) = Responses.subcube(i,0,t,i,(Jt-1),t);
    }
  }
  return Y_miss;
}


//' @title Compute item pairwise odds ratio
//' @description Based on a response matrix, calculate the item pairwise odds-ratio according do (n11*n00)/(n10*n01), where nij is the
//' number of people answering both item i and item j correctly
//' @param N An \code{int} of the sample size
//' @param J An \code{int} of the number of items
//' @param Yt An N-by-J response \code{matrix}
//' @return A J-by-J upper-triangular \code{matrix} of the item pairwise odds ratios
//' @examples 
//' OddsRatio(N,J,Y_sim)
//' @export
// [[Rcpp::export]]
arma::mat OddsRatio(unsigned int N,unsigned int J,const arma::mat& Yt){
  arma::mat M2_temp=arma::zeros<arma::mat>(J,J);
  for(unsigned int j1=0;j1<J-1;j1++){
    for(unsigned int j2=j1+1;j2<J;j2++){
      double n11 = arma::accu(Yt.col(j1)%Yt.col(j2));
      double n00 = arma::accu((1.-Yt.col(j1))%(1.-Yt.col(j2)));
      double n10 = arma::accu(Yt.col(j1)%(1.-Yt.col(j2)));
      double n01 = N-n11-n00-n10;
      M2_temp(j1,j2) = (n11*n00)/(n10*n01); 
    }
  }
  return M2_temp;
}




// [[Rcpp::export]]
int getMode(arma::vec sorted_vec, int size){
  int counter = 1;
  int max = 0;
  int mode = sorted_vec(0);
  for (int pass = 0; pass < size - 1; pass++)
  {
    if ( sorted_vec(pass) == sorted_vec(pass+1) )
    {
      counter++;
      if ( counter > max )
      {
        max = counter;
        mode = sorted_vec(pass);
      }
    } else
      counter = 1; // reset counter.
  }
  return mode;
}


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
//' J = 10
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
//' 
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
//' J = 10
//' K = 4
//' Q = random_Q(J,K)
//' Smats <- matrix(runif(Jt*K,.1,.3),Jt,K)
//' Gmats <- matrix(runif(Jt*K,.1,.3),Jt,K)
//' r_stars <- matrix(NA,Jt,K)
//' pi_stars <- numeric(Jt)
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
//' data("Spatial_Rotation")
//' Smats <- array(runif(Jt*K*(T),.1,.3),c(Jt,K,(T)))
//' Gmats <- array(runif(Jt*K*(T),.1,.3),c(Jt,K,(T)))
//' r_stars <- array(NA,c(Jt,K,T))
//' pi_stars <- matrix(NA,Jt,(T))
//' for(t in 1:T){
//'   pi_stars[,,t] <- apply(((1-Smats[,,t])^Qs[,,t]),1,prod)
//'   r_stars[,,t] <- Gmats[,,t]/(1-Smats[,,t])
//' }
//' Test_versions_sim <- sample(1:5,N,replace = T)
//' 
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
//' J = 10
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
//' data("Spatial_Rotation")
//' Svec <- runif(K,.1,.3)
//' Gvec <- runif(K,.1,.3)
//' Test_versions_sim <- sample(1:5,N,replace = T)
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
//' data("Spatial_Rotation")
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
//' data("Spatial_Rotation")
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
//' data("Spatial_Rotation")
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











// ----------------------------- MCMC Functions --------------------------------------------------------------
// Single iteration parameter updates, full Gibbs sampler, DIC computation 
// -----------------------------------------------------------------------------------------------------------



// [[Rcpp::export]]
Rcpp::List parm_update_HO(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                          arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                          const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                          const arma::mat test_order, const arma::vec Test_versions, 
                          const double theta_propose, const arma::vec deltas_propose){
  arma::cube ETA(Jt, (pow(2,K)), T);
  arma::vec CLASS_0(N);
  for(unsigned int t = 0; t<T; t++){
    ETA.slice(t) = ETAmat(K,Jt, Qs.slice(t));
  }
  
  double post_new, post_old;
  double theta_i_new;
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  
  for(unsigned int i = 0; i<N; i++){
    int test_version_i = Test_versions(i)-1;
    double theta_i = thetas(i);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      int test_block_it = test_order(test_version_i,t)-1;
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        likelihood_Y(cc) = pYit_DINA(ETA.slice(test_block_it).col(cc), response.slice(t).row(i).t(),itempars.slice(test_block_it));
      }
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      // initial time point
      if(t == 0){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % pi % ptranspost;
        probs = probs/arma::sum(probs);
        double tmp = rmultinomial(probs);
        CLASS_0(i) = tmp;
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // middle points
      if(t > 0 && t < (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probabilities
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % ptransprev % ptranspost;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // last time point
      if(t == (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probs
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % ptransprev;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
    }
    // update theta_i
    theta_i_new = R::rnorm(theta_i,theta_propose);
    
    // The prior of for theta, N(0,1)
    post_old = std::log(R::dnorm(theta_i, 0, 1, false));
    post_new = std::log(R::dnorm(theta_i_new, 0, 1, false));
    
    // multiply prior by trans prob at time t>0
    for(unsigned int t = 1; t<T; t++){
      post_old += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Jt,(t-1)));
      post_new += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i_new,Q_i,Jt,(t-1)));
    }
    ratio = exp(post_new - post_old);
    u = R::runif(0,1);
    if(u < ratio){
      thetas(i) = theta_i_new;
      accept_theta(i) = 1;
    }
  }
  
  // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  // update lambdas
  arma::vec accept_lambdas = arma::zeros<arma::vec>(lambdas.n_elem);
  for(unsigned int h = 0; h<lambdas.n_elem; h++){
    // reset tmp
    arma::vec tmp = lambdas;
    tmp(h) = R::runif((lambdas(h)-deltas_propose(h)), (lambdas(h)+deltas_propose(h)));
    if(h == 0){
      post_old = R::dnorm(lambdas(h),0,0.5,true);
      post_new = R::dnorm(tmp(h),0,0.5,true);
    }else{
      if (h == 1) {
        post_old = R::dlnorm(lambdas(h), 0, .5, 1);
        post_new = R::dlnorm(tmp(h), 0, .5, 1);
      }
      else {
        post_old = R::dlnorm(lambdas(h), -0.5, .6, 1);
        post_new = R::dlnorm(tmp(h), -0.5, .6, 1);
      }
    }
    
    for(unsigned int i = 0; i < N; i++){
      for(unsigned int t = 0; t < (T-1); t++){
        post_old += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas, thetas(i), Q_examinee[i], Jt, t));
        post_new += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          tmp, thetas(i), Q_examinee[i], Jt, t));
      }
    }
    
    ratio = exp(post_new-post_old);
    u = R::runif(0,1);
    if(u < ratio){
      lambdas = tmp;
      accept_lambdas(h) = 1;
    }
  }
  
  // update s, g, alpha, gamma for items, and save the aggregated coefficients for the posterior of phi
  double as, bs, ag, bg, pg, ps, ug, us;
  for(unsigned int block = 0; block < T; block++){
    arma::mat Res_block(N, Jt);
    arma::mat RT_block(N, Jt);
    arma::mat Q_current = Qs.slice(block);
    arma::vec Class(N);
    arma::vec eta_j(N);
    arma::mat ETA_block(N,Jt);
    for(unsigned int i = 0; i < N; i++){
      // find the time point at which i received this block
      int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(block+1)));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.row(i) = response.slice(t_star).row(i);
      arma::vec alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      
      for(unsigned int j = 0; j < Jt; j++){
        ETA_block(i,j) = ETA(j,Class(i),block);
      }
    }
    
    for(unsigned int j = 0; j < Jt; j++){
      eta_j = ETA_block.col(j);
      
      // sample the response model parameters
      us = R::runif(0, 1);
      ug = R::runif(0, 1);
      // get posterior a, b for sj and gj
      as = arma::sum(eta_j % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      bs = arma::sum(eta_j % Res_block.col(j)) + 1;
      ag = arma::sum((arma::ones<arma::vec>(N)-eta_j) % Res_block.col(j)) + 1;
      bg = arma::sum((arma::ones<arma::vec>(N)-eta_j) % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      // update g based on s on previous iteration
      pg = R::pbeta(1.0 - itempars(j,0,block), ag, bg, 1, 0);
      itempars(j,1,block) = R::qbeta(ug*pg, ag, bg, 1, 0);
      // update s based on current g
      ps = R::pbeta(1.0 - itempars(j,1,block), as, bs, 1, 0);
      itempars(j,0,block) = R::qbeta(us*ps, as, bs, 1, 0);
    }
  }
  
  
  // return the results
  return Rcpp::List::create(Rcpp::Named("accept_theta",accept_theta),
                            Rcpp::Named("accept_lambdas",accept_lambdas)
  );
}


// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_HO(const arma::cube& Response, 
                         const arma::cube& Qs, const Rcpp::List Q_examinee,
                         const arma::mat& test_order, const arma::vec& Test_versions, 
                         const double theta_propose,const arma::vec deltas_propose,
                         const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  
  //std::srand(std::time(0));//use current time as seed for random generator
  
  // initialize parameters
  arma::vec lambdas_init(4),tauvar_init(1);
  lambdas_init(0) = R::rnorm(0,1);
  lambdas_init(1) = R::runif(0,1);
  lambdas_init(2) = R::runif(0,1);
  lambdas_init(3) = R::runif(0,1);
  
  arma::vec thetas_init(N);
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  tauvar_init(0) = R::runif(1, 1.5); // initial value for the variance of taus
  
  for(unsigned int i = 0; i < N; i++){
    thetas_init(i,0) = R::rnorm(0, 1);
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  arma::cube Alphas_init = simulate_alphas_HO_sep(lambdas_init,thetas_init,Alphas_0_init,
                                                  Q_examinee, T, Jt);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  
  arma::cube itempars_init = .3 * arma::randu<arma::cube>(Jt,2,T);
  itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) =
    itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) % (1.-itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)));
  
  // Create objects for storage
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat ss(J,(chain_length-burn_in));
  arma::mat gs(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat thetas(N,(chain_length-burn_in));
  arma::mat lambdas(4,(chain_length-burn_in));
  double accept_rate_theta = 0;
  arma::vec accept_rate_lambdas = arma::zeros<arma::vec>(4);
  
  double tmburn;//,deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for (unsigned int tt = 0; tt < chain_length; tt++) {
    Rcpp::List tmp = parm_update_HO(N,Jt,K,T,Alphas_init,pi_init,lambdas_init,thetas_init,Response,
                                    itempars_init,Qs,Q_examinee,test_order,Test_versions,
                                    theta_propose,deltas_propose);
    if (tt >= burn_in) {
      tmburn = tt - burn_in;
      for (unsigned int t = 0; t < T; t++) {
        Trajectories_mat.cols(K*t, (K*(t + 1) - 1)) = Alphas_init.slice(t);
        ss.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(0);
        gs.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(1);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      thetas.col(tmburn) = thetas_init;
      lambdas.col(tmburn) = lambdas_init;
      accept_theta_vec = Rcpp::as<arma::vec>(tmp[0]);
      accept_lambdas_vec = Rcpp::as<arma::vec>(tmp[1]);
      m_accept_theta = arma::mean(accept_theta_vec);
      accept_rate_theta = (accept_rate_theta*tmburn + m_accept_theta) / (tmburn + 1.);
      accept_rate_lambdas = (accept_rate_lambdas*tmburn + accept_lambdas_vec) / (tmburn + 1.);
      
      
      
    }
    
    if (tt % 1000 == 0) {
      Rcpp::Rcout << tt << std::endl;
    }
  }
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",ss),
                            Rcpp::Named("gs",gs),
                            Rcpp::Named("pis", pis),
                            Rcpp::Named("thetas",thetas),
                            Rcpp::Named("lambdas",lambdas),
                            Rcpp::Named("accept_rate_theta",accept_rate_theta),
                            Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas)
                              // Rcpp::Named("accept_rate_tau", accept_rate_tau),
                              // Rcpp::Named("time_pp", time_pp),
                              // Rcpp::Named("res_pp", res_pp),
                              // Rcpp::Named("Deviance",Deviance),
                              // Rcpp::Named("D_DINA", Deviance_DINA),
                              // Rcpp::Named("D_tran",Deviance_tran)
  );
}












// [[Rcpp::export]]
Rcpp::List parm_update_HO_RT_sep(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                                 arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                 const arma::cube latency, arma::cube& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::vec& tauvar,
                                 const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                                 const arma::mat test_order, const arma::vec Test_versions, const int G_version,
                                 const double theta_propose, const double a_sigma_tau0, const double rate_sigma_tau0, 
                                 const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0
){
  double phi = phi_vec(0);
  double tau_sig = tauvar(0);
  arma::cube ETA(Jt, (pow(2,K)), T);
  arma::vec CLASS_0(N);
  for(unsigned int t = 0; t<T; t++){
    ETA.slice(t) = ETAmat(K,Jt, Qs.slice(t));
  }
  arma::cube J_incidence = J_incidence_cube(test_order,Qs);
  
  double post_new, post_old;
  arma::vec thetatau_i_old(2);
  arma::vec thetatau_i_new(2);
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  arma::vec accept_tau = arma::zeros<arma::vec>(N);
  
  for(unsigned int i = 0; i<N; i++){
    int test_version_i = Test_versions(i)-1;
    double theta_i = thetas(i);
    double tau_i = taus(i);
    arma::vec G_it = arma::zeros<arma::vec>(Jt);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      int test_block_it = test_order(test_version_i,t)-1;
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        likelihood_Y(cc) = pYit_DINA(ETA.slice(test_block_it).col(cc), response.slice(t).row(i).t(),itempars.slice(test_block_it));
      }
      // likelihood of RT (time dependent)
      arma::vec likelihood_L = arma::ones<arma::vec>(pow(2,K));
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      int test_block_itt;
      // initial time point
      if(t == 0){
        
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          
          // likelihood of RT
          if(G_version !=3){
            if(G_version==1 ){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =0; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
            
            
          }else{
            likelihood_L(cc)=1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % pi % ptranspost;
        probs = probs/arma::sum(probs);
        double tmp = rmultinomial(probs);
        CLASS_0(i) = tmp;
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // middle points
      if(t > 0 && t < (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probabilities
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          
          // likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =t; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
          }else{
            likelihood_L(cc)=1;
          }
          
          
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev % ptranspost;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      
      // last time point
      if(t == (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probs
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          // Likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
            }
            if(G_version==2){
              G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                     test_order,t);
            }
            
            
            
            likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                         tau_i,phi);
          }else{
            
            likelihood_L(cc) =1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
    }
    // update theta_i
    
    thetatau_i_new(0)= R::rnorm(theta_i,theta_propose);
    
    // The prior of for theta, N(0,1)
    
    post_old = std::log(R::dnorm(theta_i, 0, 1, false));
    post_new = std::log(R::dnorm(thetatau_i_new(0), 0, 1, false));
    
    // multiply prior by trans prob at time t>0
    for(unsigned int t = 1; t<T; t++){
      
      post_old += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Jt,(t-1)));
      post_new += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,thetatau_i_new(0),Q_i,Jt,(t-1)));
    }
    ratio = exp(post_new - post_old);
    u = R::runif(0,1);
    if(u < ratio){
      thetas(i) = thetatau_i_new(0);
      accept_theta(i) = 1;
    }
    
    // update tau_i, Gbbs, draw the tau_i from the posterial distribution, which is still normal
    
    double num = 0;
    double denom = 0;
    test_version_i = Test_versions(i) - 1;
    for (unsigned int t = 0; t<T; t++) {
      int	test_block_it = test_order(test_version_i, t) - 1;
      
      if (G_version == 1) {
        double class_it = arma::dot(alphas.slice(t).row(i).t(), vv);
        G_it = ETA.slice(test_block_it).col(class_it);
      }
      if (G_version == 2) {
        G_it = G2vec_efficient(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                               test_order, t);
      }
      if(G_version==3){
        
        G_it = arma::ones<arma::vec>(Jt);
        arma::vec y(Jt);
        y.fill((t+1.)/T);
        G_it =G_it % y;
      }
      
      
      for (unsigned int j = 0; j<Jt; j++) {
        //add practice*a of (i,j,block) to num and denom of phi
        num += (log(latency(i, j, t))-RT_itempars(j, 1, test_block_it)+ phi*G_it(j))*pow(RT_itempars(j, 0, test_block_it), 2);
        denom += pow(RT_itempars(j, 0, test_block_it), 2);
      }
    }
    // sample tau_i
    double mu_tau, sigma_tau;
    mu_tau = -num / (denom + (1/tau_sig));
    sigma_tau = sqrt(1. / (denom + (1/tau_sig)));
    taus(i) = R::rnorm(mu_tau, sigma_tau);
    
  }
  
  //update the variance for tau from inverse-gamma distribution
  // check this inverse
  
  double a_sigma_tau = a_sigma_tau0 + N / 2.;
  double b_sigma_tau = 1. / (rate_sigma_tau0 + arma::dot(taus.t(), taus) / 2.);
  tauvar(0) = 1. / R::rgamma(a_sigma_tau, b_sigma_tau);
  
  // // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  // update lambdas
  arma::vec accept_lambdas = arma::zeros<arma::vec>(lambdas.n_elem);
  for(unsigned int h = 0; h<lambdas.n_elem; h++){
    // reset tmp
    arma::vec tmp = lambdas;
    tmp(h) = R::runif((lambdas(h)-deltas_propose(h)), (lambdas(h)+deltas_propose(h)));
    if(h == 0){
      post_old = R::dnorm(lambdas(h),0,0.5,true);
      post_new = R::dnorm(tmp(h),0,0.5,true);
    }else{
      if (h == 1) {
        post_old = R::dlnorm(lambdas(h), 0, .5, 1);
        post_new = R::dlnorm(tmp(h), 0, .5, 1);
      }
      else {
        post_old = R::dlnorm(lambdas(h), -0.5, .6, 1);
        post_new = R::dlnorm(tmp(h), -0.5, .6, 1);
      }
    }
    
    for(unsigned int i = 0; i < N; i++){
      for(unsigned int t = 0; t < (T-1); t++){
        post_old += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas, thetas(i), Q_examinee[i], Jt, t));
        post_new += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          tmp, thetas(i), Q_examinee[i], Jt, t));
      }
    }
    
    ratio = exp(post_new-post_old);
    u = R::runif(0,1);
    if(u < ratio){
      lambdas = tmp;
      accept_lambdas(h) = 1;
    }
  }
  
  // update s, g, alpha, gamma for items, and save the aggregated coefficients for the posterior of phi
  double as, bs, ag, bg, pg, ps, ug, us;
  double a_alpha, scl_alpha, mu_gamma, sd_gamma, alpha_sqr;
  double tau_i;
  arma::cube Gs(N, Jt,T);
  for(unsigned int block = 0; block < T; block++){
    arma::mat Res_block(N, Jt);
    arma::mat RT_block(N, Jt);
    arma::mat Q_current = Qs.slice(block);
    arma::vec Class(N);
    arma::vec eta_j(N);
    arma::mat ETA_block(N,Jt);
    for(unsigned int i = 0; i < N; i++){
      // find the time point at which i received this block
      int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(block+1)));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.row(i) = response.slice(t_star).row(i);
      RT_block.row(i) = latency.slice(t_star).row(i);
      arma::vec alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      
      for(unsigned int j = 0; j < Jt; j++){
        ETA_block(i,j) = ETA(j,Class(i),block);
        if(G_version == 1){
          Gs(i,j,block) = ETA_block(i,j);
        }
        if(G_version==3){
          Gs(i,j,block)= (t_star+1.)/T;
        }
      }
      if(G_version == 2){
        Gs.slice(block).row(i) = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),
                 (Test_versions(i)-1),test_order,t_star).t();
      }
      
      
    }
    
    for(unsigned int j = 0; j < Jt; j++){
      eta_j = ETA_block.col(j);
      
      // sample the response model parameters
      us = R::runif(0, 1);
      ug = R::runif(0, 1);
      // get posterior a, b for sj and gj
      as = arma::sum(eta_j % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      bs = arma::sum(eta_j % Res_block.col(j)) + 1;
      ag = arma::sum((arma::ones<arma::vec>(N)-eta_j) % Res_block.col(j)) + 1;
      bg = arma::sum((arma::ones<arma::vec>(N)-eta_j) % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      // update g based on s on previous iteration
      pg = R::pbeta(1.0 - itempars(j,0,block), ag, bg, 1, 0);
      itempars(j,1,block) = R::qbeta(ug*pg, ag, bg, 1, 0);
      // update s based on current g
      ps = R::pbeta(1.0 - itempars(j,1,block), as, bs, 1, 0);
      itempars(j,0,block) = R::qbeta(us*ps, as, bs, 1, 0);
      
      // sample the RT model parameters
      // scl_tmp: (log(L_ij) + tau_i + phi * G_ij -gamma_j)^2
      // mu_tmp: log(L_it) + tau_i + phi * G_ij
      arma::vec scl_tmp(N);
      arma::vec mu_tmp(N);
      for(unsigned int i = 0; i < N; i++){
        tau_i = taus(i);
        scl_tmp(i) = pow((log(RT_block(i,j))+tau_i+phi*Gs(i,j,block)-RT_itempars(j,1,block)),2);
        mu_tmp(i) = log(RT_block(i,j))+tau_i+phi*Gs(i,j,block);
      }
      // update alpha_j based on previous gamma_j
      a_alpha = a_alpha0 + N/2;
      // note: the derivation we have corresponds to the rate of gamma, need to take recip for scl
      scl_alpha = 1./(rate_alpha0 + 1./2 * arma::sum(scl_tmp));
      alpha_sqr = R::rgamma(a_alpha,scl_alpha);
      RT_itempars(j,0,block) = sqrt(alpha_sqr);
      // update gamma_j based on current alpha_j
      mu_gamma = (alpha_sqr*arma::sum(mu_tmp))/(N*alpha_sqr + 1);
      sd_gamma = sqrt(1./(N*alpha_sqr + 1));
      RT_itempars(j,1,block) = R::rnorm(mu_gamma, sd_gamma);
    }
  }
  
  
  // update phi
  double num = 0;
  double denom = 0;
  unsigned int test_version_i, block_it;
  for(unsigned int i = 0; i<N; i++){
    tau_i = taus(i);
    test_version_i = Test_versions(i)-1;
    for(unsigned int t = 0; t<T; t++){
      block_it = test_order(test_version_i,t)-1;
      for(unsigned int j = 0; j<Jt; j++){
        //add practice*a of (i,j,block) to num and denom of phi
        num += pow(RT_itempars(j,0,block_it),2) * Gs(i,j,block_it)*(log(latency(i,j,t))-RT_itempars(j,1,block_it)+tau_i);
        denom += pow(RT_itempars(j,0,block_it),2)* pow(Gs(i,j,block_it),2);
      }
    }
  }
  
  double mu_phi, sigma_phi;
  mu_phi = -num/(denom+1.);
  sigma_phi = sqrt(1./(denom+1.));
  phi_vec(0) = R::rnorm(mu_phi,sigma_phi);
  
  
  
  
  
  // return the results
  return Rcpp::List::create(Rcpp::Named("accept_theta",accept_theta),
                            Rcpp::Named("accept_lambdas",accept_lambdas)
  );
}


// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_HO_RT_sep(const arma::cube& Response, const arma::cube& Latency,
                                const arma::cube& Qs, const Rcpp::List Q_examinee,
                                const arma::mat& test_order, const arma::vec& Test_versions, int G_version,
                                const double theta_propose,const arma::vec deltas_propose,
                                const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  
  //std::srand(std::time(0));//use current time as seed for random generator
  
  // initialize parameters
  arma::vec lambdas_init(4),tauvar_init(1);
  lambdas_init(0) = R::rnorm(0,1);
  lambdas_init(1) = R::runif(0,1);
  lambdas_init(2) = R::runif(0,1);
  lambdas_init(3) = R::runif(0, 1);
  
  //arma::mat Sig_init = arma::eye<arma::mat>(2,2);
  arma::mat thetatau_init(N,2);
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  tauvar_init(0) = R::runif(1, 1.5); // initial value for the variance of taus
  
  for(unsigned int i = 0; i < N; i++){
    thetatau_init(i,0) = R::rnorm(0, 1);
    thetatau_init(i,1)= R::rnorm(0,tauvar_init(0));
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  arma::vec thetas_init = thetatau_init.col(0);
  arma::vec taus_init = thetatau_init.col(1);
  arma::cube Alphas_init = simulate_alphas_HO_sep(lambdas_init,thetatau_init.col(0),Alphas_0_init,
                                                  Q_examinee, T, Jt);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec phi_init(1);
  phi_init(0) = R::runif(0,1);
  //phi_init(0) =0;
  
  arma::cube itempars_init = .3 * arma::randu<arma::cube>(Jt,2,T);
  itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) =
    itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) % (1.-itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)));
  arma::cube RT_itempars_init(Jt,2,T);
  RT_itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)) = 2.+2.*arma::randu<arma::cube>(Jt,1,T);
  RT_itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) = arma::randn<arma::cube>(Jt,1,T); // why take exponetial part?
  
  //double p = 3.;
  //
  // Create objects for storage
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat ss(J,(chain_length-burn_in));
  arma::mat gs(J,(chain_length-burn_in));
  arma::mat RT_as(J,(chain_length-burn_in));
  arma::mat RT_gammas(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat thetas(N,(chain_length-burn_in));
  arma::mat taus(N,(chain_length-burn_in));
  arma::mat lambdas(4,(chain_length-burn_in));
  arma::vec phis(chain_length-burn_in);
  //arma::cube Sigs(2,2,(chain_length-burn_in));
  arma::vec tauvar(chain_length - burn_in);
  double accept_rate_theta = 0;
  //double accept_rate_tau =0;
  arma::vec accept_rate_lambdas = arma::zeros<arma::vec>(4);
  // arma::cube time_pp(N,T,(chain_length-burn_in));
  // arma::cube res_pp(N,T,(chain_length-burn_in));
  // arma::vec Deviance(chain_length-burn_in);
  // arma::vec Deviance_DINA(chain_length-burn_in);
  // arma::vec Deviance_RT(chain_length-burn_in);
  // arma::vec Deviance_tran(chain_length-burn_in);
  
  double tmburn;//,deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec,accept_tau_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for (unsigned int tt = 0; tt < chain_length; tt++) {
    Rcpp::List tmp = parm_update_HO_RT_sep(N, Jt, K, T, Alphas_init, pi_init, lambdas_init,
                                           thetas_init, Latency, RT_itempars_init, taus_init,
                                           phi_init, tauvar_init, Response, itempars_init, Qs,
                                           Q_examinee, test_order, Test_versions, G_version,
                                           theta_propose, 2.5, 1., deltas_propose, 1., 1.);
    if (tt >= burn_in) {
      tmburn = tt - burn_in;
      for (unsigned int t = 0; t < T; t++) {
        Trajectories_mat.cols(K*t, (K*(t + 1) - 1)) = Alphas_init.slice(t);
        ss.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(0);
        gs.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(1);
        RT_as.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(0);
        RT_gammas.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(1);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      thetas.col(tmburn) = thetas_init;
      taus.col(tmburn) = taus_init;
      lambdas.col(tmburn) = lambdas_init;
      phis(tmburn) = phi_init(0);
      tauvar(tmburn) = tauvar_init(0);
      
      accept_theta_vec = Rcpp::as<arma::vec>(tmp[0]);
      accept_lambdas_vec = Rcpp::as<arma::vec>(tmp[1]);
      m_accept_theta = arma::mean(accept_theta_vec);
      accept_rate_theta = (accept_rate_theta*tmburn + m_accept_theta) / (tmburn + 1.);
      accept_rate_lambdas = (accept_rate_lambdas*tmburn + accept_lambdas_vec) / (tmburn + 1.);
      // ETA = Rcpp::as<arma::cube>(tmp[2]);
      // J_incidence = Rcpp::as<arma::cube>(tmp[3]);
      // 
      // //Deviance at this iteration (We use the joint distribution)
      // 
      // double tran=0, response=0, time=0;
      // arma::vec G_it = arma::zeros<arma::vec>(Jt);
      // arma::vec vv = bijectionvector(K);
      // 
      // for (unsigned int i = 0; i < N; i++) {
      //   int test_version_i = Test_versions(i) - 1;
      //   for (unsigned int t = 0; t < T; t++) {
      //     
      //     // The transition model part
      //     if (t < (T - 1)) {
      //       tran += std::log(pTran_HO_sep(Alphas_init.slice(t).row(i).t(),
      //                              Alphas_init.slice(t + 1).row(i).t(),
      //                              lambdas_init, thetas_init(i), Q_examinee[i], Jt, t));
      //     }
      //     // The log likelihood from response time model
      //     int test_block_it = test_order(test_version_i, t) - 1;
      //     double class_it = arma::dot(Alphas_init.slice(t).row(i).t(), vv);
      //     if (G_version == 1) {
      //       G_it = ETA.slice(test_block_it).col(class_it);
      //     }
      //     if (G_version == 2) {
      //       G_it = G2vec_efficient(ETA, J_incidence, Alphas_init.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
      //                              test_order, t);
      //     }
      //     if(G_version==3){
      //       
      //       G_it = arma::ones<arma::vec>(Jt);
      //       arma::vec y(Jt);y.fill((t+1.)/T);
      //       G_it =G_it % y;
      //     }
      //     
      //     time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), RT_itempars_init.slice(test_block_it),
      //                           taus_init(i), phis(tmburn)));
      //     
      //     // The loglikelihood from the DINA
      //     
      //     response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), itempars_init.slice(test_block_it)));
      //     if (t == 0) {
      //       response += std::log(pi_init(class_it));
      //     }
      //     
      //   }
      //   
      //   
      // }
      // 
      // // Calculate the deviance in this iteration 
      // 
      // deviance = response + time + tran;
      // Deviance(tmburn)=-2*deviance;
      // Deviance_DINA(tmburn)=-2*response;
      // Deviance_RT(tmburn)=-2*(time);
      // Deviance_tran(tmburn)=-2*tran;
      // 
      // // ppp
      // // Generate response time
      // arma::cube L_model = sim_RT(Alphas_init, RT_itempars_init, Qs, taus_init, phi_init(0), ETA, G_version,
      //                             test_order, Test_versions);
      // 
      // //Generate the response from DINA model
      // // store the total response time for each block, the total score of each person on each block;
      // arma::cube Res_model = simDINA(Alphas_init, itempars_init, ETA, test_order, Test_versions);
      // 
      // arma::mat L_it_model(N, T), DINA_it_model(N, T);
      // for (unsigned int t = 0; t < T; t++) {
      //   L_it_model.col(t) = arma::sum(L_model.slice(t), 1);
      //   DINA_it_model.col(t) = arma::sum(Res_model.slice(t), 1);
      // }
      // 
      // time_pp.slice(tmburn) = L_it_model;
      // res_pp.slice(tmburn) = DINA_it_model;
      // 
      
      
      
      
    }
    if (tt % 1000 == 0) {
      Rcpp::Rcout << tt << std::endl;
    }
    
  }
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",ss),
                            Rcpp::Named("gs",gs),
                            Rcpp::Named("as",RT_as),
                            Rcpp::Named("gammas",RT_gammas),
                            Rcpp::Named("pis", pis),
                            Rcpp::Named("thetas",thetas),
                            Rcpp::Named("taus",taus),
                            Rcpp::Named("lambdas",lambdas),
                            Rcpp::Named("phis",phis),
                            Rcpp::Named("tauvar", tauvar),
                            Rcpp::Named("accept_rate_theta",accept_rate_theta),
                            Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas)
                              // Rcpp::Named("accept_rate_tau", accept_rate_tau),
                              // Rcpp::Named("time_pp", time_pp),
                              // Rcpp::Named("res_pp", res_pp),
                              // Rcpp::Named("Deviance",Deviance),
                              // Rcpp::Named("D_DINA", Deviance_DINA),
                              // Rcpp::Named("D_RT", Deviance_RT),
                              // Rcpp::Named("D_tran",Deviance_tran)
  );
}





// [[Rcpp::export]]
Rcpp::List parm_update_HO_RT_joint(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                                   arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                   const arma::cube latency, arma::cube& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::mat& Sig,
                                   const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                                   const arma::mat test_order, const arma::vec Test_versions, const int G_version,
                                   const double sig_theta_propose, const arma::mat S, double p,
                                   const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0
){
  double phi = phi_vec(0);
  arma::vec CLASS_0(N);
  arma::cube ETA(Jt, (pow(2,K)), T);
  for(unsigned int t = 0; t<T; t++){
    ETA.slice(t) = ETAmat(K,Jt, Qs.slice(t));
  }
  arma::cube J_incidence = J_incidence_cube(test_order,Qs);
  
  double post_new, post_old;
  double theta_new;
  arma::vec thetatau_i_old(2);
  arma::vec thetatau_i_new(2);
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  arma::vec accept_tau = arma::zeros<arma::vec>(N);  
  for(unsigned int i = 0; i<N; i++){
    int test_version_i = Test_versions(i)-1;
    double theta_i = thetas(i);
    double tau_i = taus(i);
    arma::vec G_it = arma::zeros<arma::vec>(Jt);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      int test_block_it = test_order(test_version_i,t)-1;
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        likelihood_Y(cc) = pYit_DINA(ETA.slice(test_block_it).col(cc), response.slice(t).row(i).t(),itempars.slice(test_block_it));
      }
      // likelihood of RT (time dependent)
      arma::vec likelihood_L = arma::ones<arma::vec>(pow(2,K));
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      int test_block_itt;
      // initial time point
      if(t == 0){
        
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_joint(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          
          // likelihood of RT
          if(G_version !=3){
            if(G_version==1 ){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =0; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
            
            
          }else{
            likelihood_L(cc)=1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % pi % ptranspost;
        probs = probs/arma::sum(probs);
        double tmp = rmultinomial(probs);
        CLASS_0(i) = tmp;
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // middle points
      if(t > 0 && t < (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probabilities
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_joint(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_joint(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          
          // likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =t; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
          }else{
            likelihood_L(cc)=1;
          }
          
          
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev % ptranspost;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      
      // last time point
      if(t == (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probs
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_joint(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          // Likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
            }
            if(G_version==2){
              G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                     test_order,t);
            }
            
            
            
            likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                         tau_i,phi);
          }else{
            
            likelihood_L(cc) =1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
    }
    // update theta_i, tau_i
    thetatau_i_old(0) = theta_i;
    thetatau_i_old(1) = tau_i;
    
    theta_new = R::rnorm(theta_i,sig_theta_propose);
    thetatau_i_new = thetatau_i_old;
    thetatau_i_new(0) = theta_new;
    post_old = std::log(dmvnrm(thetatau_i_old,arma::zeros<arma::vec>(2),Sig));
    post_new = std::log(dmvnrm(thetatau_i_new,arma::zeros<arma::vec>(2),Sig));
    for(unsigned int t = 1; t<T; t++){
      post_old += std::log(pTran_HO_joint(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Jt,(t-1)));
      post_new += std::log(pTran_HO_joint(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_new,Q_i,Jt,(t-1)));
    }
    ratio = exp(post_new - post_old);
    u = R::runif(0,1);
    if(u < ratio){
      thetas(i) = thetatau_i_new(0);
      thetatau_i_old = thetatau_i_new;
      accept_theta(i) = 1;
    }
    
    // update tau_i, Gbbs, draw the tau_i from the posterial distribution, which is still normal
    
    double num = 0;
    double denom = 0;
    test_version_i = Test_versions(i) - 1;
    for (unsigned int t = 0; t<T; t++) {
      int	test_block_it = test_order(test_version_i, t) - 1;
      
      if (G_version == 1) {
        double class_it = arma::dot(alphas.slice(t).row(i).t(), vv);
        G_it = ETA.slice(test_block_it).col(class_it);
      }
      if (G_version == 2) {
        G_it = G2vec_efficient(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                               test_order, t);
      }
      if(G_version==3){
        
        G_it = arma::ones<arma::vec>(Jt);
        arma::vec y(Jt);y.fill((t+1.)/T);
        G_it =G_it % y;
      }
      for (unsigned int j = 0; j<Jt; j++) {
        //add practice*a of (i,j,block) to num and denom of phi
        num += (log(latency(i, j, t))-RT_itempars(j, 1, test_block_it)+ phi*G_it(j))*pow(RT_itempars(j, 0, test_block_it), 2);
        denom += pow(RT_itempars(j, 0, test_block_it), 2);
      }
    }
    // sample tau_i
    double mu_tau, sigma_tau;
    mu_tau = -num / (denom + (1/Sig(1,1)));
    sigma_tau = sqrt(1. / (denom + (1/Sig(1,1))));
    taus(i) = R::rnorm(mu_tau, sigma_tau);
  }
  
  // update Sigma for thetatau
  arma::mat thetatau_mat(N,2);
  thetatau_mat.col(0) = thetas;
  thetatau_mat.col(1) = taus;
  arma::mat S_star = thetatau_mat.t() * thetatau_mat + S;
  unsigned int p_star = p + N;
  Sig = rinvwish(p_star,S_star);
  
  // // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  // update lambdas
  arma::vec accept_lambdas = arma::zeros<arma::vec>(3);
  for(unsigned int h = 0; h<lambdas.n_elem; h++){
    // reset tmp
    arma::vec tmp = lambdas;
    tmp(h) = R::runif((lambdas(h)-deltas_propose(h)), (lambdas(h)+deltas_propose(h)));
    if(h == 0){
      post_old = R::dnorm4(lambdas(h),0,0.5,1);
      post_new = R::dnorm4(tmp(h),0,0.5,1);
    }
    else {
      
      post_old = R::dlnorm(lambdas(h), -0.5, .6, 1);
      post_new = R::dlnorm(tmp(h), -0.5, .6, 1);
      
      
    }
    
    for(unsigned int i = 0; i < N; i++){
      for(unsigned int t = 0; t < (T-1); t++){
        post_old += std::log(pTran_HO_joint(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            lambdas, thetas(i), Q_examinee[i], Jt, t));
        post_new += std::log(pTran_HO_joint(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            tmp, thetas(i), Q_examinee[i], Jt, t));
      }
    }
    
    ratio = exp(post_new-post_old);
    u = R::runif(0,1);
    if(u < ratio){
      lambdas = tmp;
      accept_lambdas(h) = 1;
    }
  }
  
  // update s, g, alpha, gamma for items, and save the aggregated coefficients for the posterior of phi
  double as, bs, ag, bg, pg, ps, ug, us;
  double a_alpha, scl_alpha, mu_gamma, sd_gamma, alpha_sqr;
  double tau_i;
  arma::cube Gs(N, Jt,T);
  for(unsigned int block = 0; block < T; block++){
    arma::mat Res_block(N, Jt);
    arma::mat RT_block(N, Jt);
    arma::mat Q_current = Qs.slice(block);
    arma::vec Class(N);
    arma::vec eta_j(N);
    arma::mat ETA_block(N,Jt);
    for(unsigned int i = 0; i < N; i++){
      // find the time point at which i received this block
      int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(block+1)));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.row(i) = response.slice(t_star).row(i);
      RT_block.row(i) = latency.slice(t_star).row(i);
      arma::vec alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      
      for(unsigned int j = 0; j < Jt; j++){
        ETA_block(i,j) = ETA(j,Class(i),block);
        if(G_version == 1){
          Gs(i,j,block) = ETA_block(i,j);
        }
        
        if(G_version==3){
          Gs(i,j,block)= (t_star+1.)/(T);
        }
        
      }
      if(G_version == 2){
        Gs.slice(block).row(i) = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),
                 (Test_versions(i)-1),test_order,t_star).t();
      }
    }
    
    for(unsigned int j = 0; j < Jt; j++){
      eta_j = ETA_block.col(j);
      
      // sample the response model parameters
      us = R::runif(0, 1);
      ug = R::runif(0, 1);
      // get posterior a, b for sj and gj
      as = arma::sum(eta_j % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      bs = arma::sum(eta_j % Res_block.col(j)) + 1;
      ag = arma::sum((arma::ones<arma::vec>(N)-eta_j) % Res_block.col(j)) + 1;
      bg = arma::sum((arma::ones<arma::vec>(N)-eta_j) % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      // update g based on s on previous iteration
      pg = R::pbeta(1.0 - itempars(j,0,block), ag, bg, 1, 0);
      itempars(j,1,block) = R::qbeta(ug*pg, ag, bg, 1, 0);
      // update s based on current g
      ps = R::pbeta(1.0 - itempars(j,1,block), as, bs, 1, 0);
      itempars(j,0,block) = R::qbeta(us*ps, as, bs, 1, 0);
      
      // sample the RT model parameters
      // scl_tmp: (log(L_ij) + tau_i + phi * G_ij -gamma_j)^2
      // mu_tmp: log(L_it) + tau_i + phi * G_ij
      arma::vec scl_tmp(N);
      arma::vec mu_tmp(N);
      for(unsigned int i = 0; i < N; i++){
        tau_i = taus(i);
        scl_tmp(i) = pow((log(RT_block(i,j))+tau_i+phi*Gs(i,j,block)-RT_itempars(j,1,block)),2);
        mu_tmp(i) = log(RT_block(i,j))+tau_i+phi*Gs(i,j,block);
      }
      // update alpha_j based on previous gamma_j
      a_alpha = a_alpha0 + N/2;
      // note: the derivation we have corresponds to the rate of gamma, need to take recip for scl
      scl_alpha = 1./(rate_alpha0 + 1./2 * arma::sum(scl_tmp));
      alpha_sqr = R::rgamma(a_alpha,scl_alpha);
      RT_itempars(j,0,block) = sqrt(alpha_sqr);
      // update gamma_j based on current alpha_j
      mu_gamma = (alpha_sqr*arma::sum(mu_tmp))/(N*alpha_sqr + 1);
      sd_gamma = sqrt(1./(N*alpha_sqr + 1));
      RT_itempars(j,1,block) = R::rnorm(mu_gamma, sd_gamma);
    }
  }
  
  
  // update phi
  double num = 0;
  double denom = 0;
  unsigned int test_version_i, block_it;
  for(unsigned int i = 0; i<N; i++){
    tau_i = taus(i);
    test_version_i = Test_versions(i)-1;
    for(unsigned int t = 0; t<T; t++){
      block_it = test_order(test_version_i,t)-1;
      for(unsigned int j = 0; j<Jt; j++){
        // add practice*a of (i,j,block) to num and denom of phi
        num += pow(RT_itempars(j,0,block_it),2) * Gs(i,j,block_it)*(log(latency(i,j,t))-RT_itempars(j,1,block_it)+tau_i);
        denom += pow(RT_itempars(j,0,block_it),2)* pow(Gs(i,j,block_it),2);
      }
    }
  }
  
  double mu_phi, sigma_phi;
  mu_phi = -num/(denom+1.);
  sigma_phi = sqrt(1./(denom+1.));
  phi_vec(0) = R::rnorm(mu_phi,sigma_phi);
  return Rcpp::List::create(Rcpp::Named("accept_theta",accept_theta),
                            Rcpp::Named("accept_lambdas",accept_lambdas)
  );
}


// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_HO_RT_joint(const arma::cube& Response, const arma::cube& Latency,
                                  const arma::cube& Qs, const Rcpp::List Q_examinee,
                                  const arma::mat& test_order, const arma::vec& Test_versions, int G_version,
                                  const double sig_theta_propose, const arma::vec deltas_propose,
                                  const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  // initialize parameters
  arma::vec lambdas_init(3);
  lambdas_init(0) = R::rnorm(0,1);
  lambdas_init(1) = R::runif(0,1);
  lambdas_init(2) = R::runif(0,1);
  
  arma::mat Sig_init = arma::eye<arma::mat>(2,2);
  arma::mat thetatau_init(N,2);
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  for(unsigned int i = 0; i < N; i++){
    thetatau_init.row(i) = rmvnrm(arma::zeros<arma::vec>(2), Sig_init).t();
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  arma::vec thetas_init = thetatau_init.col(0);
  arma::vec taus_init = thetatau_init.col(1);
  arma::cube Alphas_init = simulate_alphas_HO_joint(lambdas_init,thetatau_init.col(0),Alphas_0_init,
                                                    Q_examinee, T, Jt);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec phi_init(1);
  phi_init(0) = R::runif(0,1);
  
  arma::cube itempars_init = .3 * arma::randu<arma::cube>(Jt,2,T);
  itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) =
    itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) % (1.-itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)));
  arma::cube RT_itempars_init(Jt,2,T);
  RT_itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)) = 2.+2.*arma::randu<arma::cube>(Jt,1,T);
  RT_itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) = arma::randn<arma::cube>(Jt,1,T)*.5+3.45;
  
  
  
  arma::mat S = arma::eye<arma::mat>(2,2);
  double p = 3.;
  //
  // Create objects for storage
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat ss(J,(chain_length-burn_in));
  arma::mat gs(J,(chain_length-burn_in));
  arma::mat RT_as(J,(chain_length-burn_in));
  arma::mat RT_gammas(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat thetas(N,(chain_length-burn_in));
  arma::mat taus(N,(chain_length-burn_in));
  arma::mat lambdas(3,(chain_length-burn_in));
  arma::vec phis(chain_length-burn_in);
  arma::cube Sigs(2,2,(chain_length-burn_in));
  double accept_rate_theta = 0;
  arma::vec accept_rate_lambdas = arma::zeros<arma::vec>(3);
  // arma::cube time_pp(N,T,(chain_length-burn_in));
  // arma::cube res_pp(N,T,(chain_length-burn_in));
  // arma::vec Deviance(chain_length-burn_in);
  // arma::vec Deviance_DINA(chain_length-burn_in);
  // arma::vec Deviance_RT(chain_length-burn_in);
  // arma::vec Deviance_tran(chain_length-burn_in);
  // arma::vec Deviance_joint(chain_length-burn_in);
  
  double tmburn;
  // double deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for(unsigned int tt = 0; tt < chain_length; tt ++){
    Rcpp::List tmp = parm_update_HO_RT_joint(N, Jt, K, T, Alphas_init, pi_init, lambdas_init,
                                             thetas_init, Latency, RT_itempars_init, taus_init,
                                             phi_init, Sig_init, Response, itempars_init, Qs,
                                             Q_examinee, test_order, Test_versions, G_version,
                                             sig_theta_propose, S, p, deltas_propose, 1., 1.);
    if(tt>=burn_in){
      tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
        ss.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(0);
        gs.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(1);
        RT_as.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(0);
        RT_gammas.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(1);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      thetas.col(tmburn) = thetas_init;
      taus.col(tmburn) = taus_init;
      lambdas.col(tmburn) = lambdas_init;
      phis(tmburn) = phi_init(0);
      Sigs.slice(tmburn) = Sig_init;
      
      accept_theta_vec = Rcpp::as<arma::vec>(tmp[0]);
      accept_lambdas_vec = Rcpp::as<arma::vec>(tmp[1]);
      m_accept_theta = arma::mean(accept_theta_vec);
      accept_rate_theta = (accept_rate_theta*tmburn+m_accept_theta)/(tmburn+1.);
      accept_rate_lambdas = (accept_rate_lambdas*tmburn + accept_lambdas_vec)/(tmburn+1.);
      
      // ETA = Rcpp::as<arma::cube>(tmp[2]);
      // J_incidence = Rcpp::as<arma::cube>(tmp[3]);
      
      // //Deviance at this iteration (We use the joint distribution)
      // 
      // double tran=0, response=0, time=0,joint=0;
      // arma::vec G_it = arma::zeros<arma::vec>(Jt);
      // arma::vec vv = bijectionvector(K);
      // arma::vec thetatau(2);
      // 
      // for (unsigned int i = 0; i < N; i++) {
      //   int test_version_i = Test_versions(i) - 1;
      //   
      //    // the joint distribution of theta and tau
      //     thetatau(0)=thetas_init(i);
      //     thetatau(1)=taus_init(i);
      //     
      //     joint +=  std::log (dmvnrm(thetatau,arma::zeros<arma::vec>(2),Sig_init));
      //  
      //   
      //   for (unsigned int t = 0; t < T; t++) {
      //     
      //     // The transition model part
      //     if (t < (T - 1)) {
      //       tran += std::log(pTran(Alphas_init.slice(t).row(i).t(),
      //                              Alphas_init.slice(t + 1).row(i).t(),
      //                              lambdas_init, thetas_init(i), Q_examinee[i], Jt, t));
      //     }
      //     
      //     // The log likelihood from response time model
      //     int test_block_it = test_order(test_version_i, t) - 1;
      //     double class_it = arma::dot(Alphas_init.slice(t).row(i).t(), vv);
      //     if (G_version == 1) {
      //       G_it = ETA.slice(test_block_it).col(class_it);
      //     }
      //     if (G_version == 2) {
      //       G_it = G2vec_efficient(ETA, J_incidence, Alphas_init.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
      //                              test_order, t);
      //     }
      //     
      //     if(G_version==3){
      //       
      //       G_it = arma::ones<arma::vec>(Jt);
      //       arma::vec y(Jt);y.fill((t+1.)/T);
      //       G_it =G_it % y;
      //     }
      //     
      //     time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), RT_itempars_init.slice(test_block_it),
      //                           taus_init(i), phis(tmburn)));
      //     
      //     // The loglikelihood from the DINA
      //     
      //     response += std::log(pYit(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), itempars_init.slice(test_block_it)));
      //     if (t == 0) {
      //       response += std::log(pi_init(class_it));
      //     }
      //     
      //   }
      //   
      //   
      // }
      // 
      // // Calculate the deviance in this iteration 
      // 
      // deviance = response + time + tran+joint;
      // Deviance(tmburn)=-2*deviance;
      // Deviance_DINA(tmburn)=-2*response;
      // Deviance_RT(tmburn)=-2*(time);
      // Deviance_tran(tmburn)=-2*tran;
      // Deviance_joint(tmburn)=-2*joint;
      
      //         // ppp
      // 		  // Generate response time
      // 		  arma::cube L_model = sim_RT(Alphas_init, RT_itempars_init, Qs, taus_init, phi_init(0), ETA, G_version,
      // 			  test_order, Test_versions);
      // 
      // 		  //Generate the response from DINA model
      // 		  // store the total response time for each block, the total score of each person on each block;
      // 		  arma::cube Res_model = simDINA(Alphas_init, itempars_init, ETA, test_order, Test_versions);
      // 
      // 		  arma::mat L_it_model(N, T), DINA_it_model(N, T);
      // 		  for (unsigned int t = 0; t < T; t++) {
      // 			  L_it_model.col(t) = arma::sum(L_model.slice(t), 1);
      // 			  DINA_it_model.col(t) = arma::sum(Res_model.slice(t), 1);
      // 		  }
      // 
      // 		  time_pp.slice(tmburn) = L_it_model;
      // 		  res_pp.slice(tmburn) = DINA_it_model;
      
      
      
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",ss),
                            Rcpp::Named("gs",gs),
                            Rcpp::Named("as",RT_as),
                            Rcpp::Named("gammas",RT_gammas),
                            Rcpp::Named("pis", pis),
                            Rcpp::Named("thetas",thetas),
                            Rcpp::Named("taus",taus),
                            Rcpp::Named("lambdas",lambdas),
                            Rcpp::Named("phis",phis),
                            Rcpp::Named("Sigs",Sigs),
                            Rcpp::Named("accept_rate_theta",accept_rate_theta),
                            Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas)
                              // Rcpp::Named("time_pp", time_pp),
                              // Rcpp::Named("res_pp", res_pp),
                              // Rcpp::Named("Deviance",Deviance),
                              // Rcpp::Named("D_DINA", Deviance_DINA),
                              // Rcpp::Named("D_RT", Deviance_RT),
                              // Rcpp::Named("D_tran",Deviance_tran),
                              // Rcpp::Named("D_joint",Deviance_joint)
  );
}



//[[Rcpp::export]]
void parm_update_rRUM(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                      arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, 
                      arma::cube& r_stars, arma::mat& pi_stars, const arma::cube Qs, 
                      const arma::cube& responses, arma::cube& X_ijk, arma::cube& Smats, arma::cube& Gmats,
                      const arma::mat& test_order,const arma::vec& Test_versions, const arma::vec& dirich_prior){
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj,prodXijk,pi_ijk,aik,u,compare;
  double pi_ik,aik_nmrtr_k,aik_dnmntr_k,c_aik_1,c_aik_0,ptranspost_1,ptranspost_0,ptransprev_1,ptransprev_0;
  arma::vec aik_nmrtr(K);
  arma::vec aik_dnmntr(K);
  double D_bar = 0;
  arma::mat Classes(N,(T));
  
  // update X
  for(unsigned int i=1;i<N;i++){
    unsigned int test_version_it = Test_versions(i)-1;
    for(unsigned int t=0; t<(T); t++){
      unsigned int block = test_order(test_version_it,t)-1;
      arma::vec pi_star_it = pi_stars.col(block);
      arma::mat r_star_it = r_stars.slice(block);
      arma::mat Q_it = Qs.slice(block);
      
      arma::vec alpha_i =(alphas.slice(t).row(i)).t();
      arma::vec Yi =(responses.slice(t).row(i)).t();
      arma::vec ui = arma::randu<arma::vec>(K);
      aik_nmrtr    = arma::ones<arma::vec>(K);
      aik_dnmntr   = arma::ones<arma::vec>(K);
      
      for(unsigned int j=0;j<Jt;j++){
        double Yij = Yi(j);
        // Note that the Xijk cube is indexed from j = 1 to Jk*(T)
        unsigned int j_star = block*Jt+j;
        arma::vec Xij = X_ijk.tube(i,j_star); 
        arma::uvec task_ij = find(Q_it.row(j) == 1);
        
        for(unsigned int k = 0;k<task_ij.n_elem ;k++){
          kj = task_ij(k);
          aik = alpha_i(kj);
          Xij(kj) = 1;
          prodXijk = prod(Xij(task_ij));
          u = R::runif(0.0,1.0);
          pi_ijk = (1.0-prodXijk)*(aik*(1.0-Smats(j,kj,block)) + (1.0-aik)*Gmats(j,kj,block) );
          compare=(pi_ijk>u);
          Xij(kj)=(1.0-Yij)*compare + Yij;
          
          aik_nmrtr(kj) = ( Xij(kj)*(1.0-Smats(j,kj,block)) + (1.0-Xij(kj))*Smats(j,kj,block) )*aik_nmrtr(kj);
          aik_dnmntr(kj) = ( Xij(kj)*Gmats(j,kj,block) + (1.0-Xij(kj))*(1.0-Gmats(j,kj,block)) )*aik_dnmntr(kj);
        }
        X_ijk.tube(i,j_star) = Xij;
      }
      // Rcpp::Rcout<<aik_nmrtr<<std::endl;
      // Rcpp::Rcout<<aik_dnmntr<<std::endl;
      
      //Update alpha_ikt
      for(unsigned int k=0;k<K;k++){
        arma::vec alpha_i_1 = alpha_i;
        alpha_i_1(k) = 1.0;
        c_aik_1 = (arma::conv_to< double >::from( alpha_i_1.t()*bijectionvector(K) ));
        arma::vec alpha_i_0 = alpha_i;
        alpha_i_0(k) = 0.0;
        c_aik_0 = (arma::conv_to< double >::from( alpha_i_0.t()*bijectionvector(K) ));
        // Rcpp::Rcout<<alpha_i_1<<std::endl;
        // Rcpp::Rcout<<alpha_i_0<<std::endl;
        
        // initial time point
        if(t == 0){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          // Rcpp::Rcout<<alpha_post<<std::endl;
          // 
          // Rcpp::Rcout<<ptranspost_1<<std::endl;
          // Rcpp::Rcout<<ptranspost_0<<std::endl;
          aik_nmrtr_k = aik_nmrtr(k)*pi(c_aik_1)*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*pi(c_aik_0)*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        
        // middle points
        if(t > 0 && t < (T-1)){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        // last time point
        if(t == (T-1)){
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
      }
      alphas.slice(t).row(i) = alpha_i.t();
      // Get DIC
      D_bar += log(pYit_rRUM(alpha_i,Yi,pi_star_it,r_star_it,Q_it));
    }
  }
  
  for(unsigned int t = 0; t<(T); t++){
    Classes.col(t) = alphas.slice(t) * bijectionvector(K);
  }
  
  arma::vec CLASS_0 = alphas.slice(0) * bijectionvector(K);
  
  // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +dirich_prior;
  pi = rDirichlet(deltatilde);
  
  // update item parameters
  //update Smat and Gmat
  double pg,ps,ug,us,gjk,sjk;
  
  for(unsigned int j_star=0;j_star<(Jt*(T));j_star++){
    unsigned int test_version_j = floor(j_star/Jt);
    unsigned int j = j_star % Jt;
    arma::uvec task_ij = find(Qs.slice(test_version_j).row(j) == 1);
    arma::mat Xj = X_ijk.tube(0,j_star,N-1,j_star);
    double pistar_temp =1.0;
    
    arma::mat alpha(N,K);
    for(unsigned int i = 0; i<N; i++){
      unsigned int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(test_version_j+1)));
      alpha.row(i) = alphas.slice(t_star).row(i);
    }
    
    for(unsigned int k = 0;k<task_ij.n_elem ;k++){
      kj = task_ij(k);
      arma::vec Xjk = Xj.col(kj);
      arma::vec ak = alpha.col(kj);
      
      double Sumalphak =  (arma::conv_to< double >::from(ak.t() * ak));
      double SumXjk = (arma::conv_to< double >::from(Xjk.t() * Xjk));
      double SumXjkalphak = (arma::conv_to< double >::from(Xjk.t() * ak));
      double bsk = SumXjkalphak ;
      double ask = Sumalphak - SumXjkalphak ;
      double agk = SumXjk - SumXjkalphak ;
      double bgk = N - SumXjk - Sumalphak + SumXjkalphak ;
      ug = R::runif(0.0,1.0);
      us = R::runif(0.0,1.0);
      
      //draw g conditoned upon s_t-1
      pg = R::pbeta(1.0-Smats(j,kj,test_version_j),agk+1.0,bgk+1.0,1,0);
      gjk = R::qbeta(ug*pg,agk+1.0,bgk+1.0,1,0);
      //draw s conditoned upon g
      ps = R::pbeta(1.0-gjk,ask+1.0,bsk+1.0,1,0);
      sjk = R::qbeta(us*ps,ask+1.0,bsk+1.0,1,0);
      
      Gmats(j,kj,test_version_j) = gjk;
      Smats(j,kj,test_version_j) = sjk;
      
      r_stars(j,kj,test_version_j) = gjk/(1.0 - sjk);//compute rstarjk
      pistar_temp = (1.0-sjk)*pistar_temp;//compute pistarj
    }
    pi_stars(j,test_version_j) = pistar_temp;
  }
  
  // Update transition probabilities
  for(unsigned int k = 0; k<K; k++){
    double a_tau, b_tau;
    a_tau = b_tau = 1;
    arma::uvec prereqs = arma::find(R.row(k)==1);
    if(prereqs.n_elem==0){
      for(unsigned int t =0; t<(T-1); t++){
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    if(prereqs.n_elem>0){
      for(unsigned int t=0; t<(T-1); t++){
        arma::mat a_prereqs = alphas.slice(t+1).cols(prereqs);
        arma::vec a_preq_1 = arma::prod(a_prereqs,1);
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0 && a_preq_1 == 1);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    taus(k) = R::rbeta((a_tau+1), (b_tau+1));
  }
  
}


// [[Rcpp::export]]
Rcpp::List Gibbs_rRUM_indept(const arma::cube& Response, const arma::cube& Qs, const arma::mat& R,
                             const arma::mat& test_order, const arma::vec& Test_versions,
                             const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  // initialize parameters
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec dirich_prior = arma::ones<arma::vec>(nClass);
  for(unsigned int cc = 0; cc < nClass; cc++){
    arma::vec alpha_cc = inv_bijectionvector(K,cc);
    for(unsigned int k = 0; k<K; k++){
      arma::uvec prereqs = arma::find(R.row(k)==1);
      if(prereqs.n_elem==0){
        if(alpha_cc(k)==1 && arma::prod(alpha_cc(prereqs))==0){
          pi_init(cc) = 0;
          dirich_prior(cc) = 0;
        }
      }
    }
  }
  
  pi_init = pi_init/arma::sum(pi_init);
  
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  for(unsigned int i = 0; i < N; i++){
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  
  arma::vec taus_init = .5*arma::ones<arma::vec>(K);
  
  arma::cube Alphas_init = simulate_alphas_indept(taus_init,Alphas_0_init,T,R);
  
  arma::cube r_stars_init = .5 + .2*arma::randu<arma::cube>(Jt,K,T);
  arma::mat pi_stars_init = .7 + .2*arma::randu<arma::mat>(Jt,T);
  
  arma::cube Smats_init = arma::randu<arma::cube>(Jt,K,T);
  arma::cube Gmats_init = arma::randu<arma::cube>(Jt,K,T) % (1-Smats_init);
  
  arma::cube X = arma::ones<arma::cube>(N,(Jt*T),K);
  
  
  // Create objects for storage 
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::cube r_stars(J,K,(chain_length-burn_in));
  arma::mat pi_stars(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat taus(K,(chain_length-burn_in));
  
  arma::mat Trajectories_mat(N,(K*T));
  arma::vec vv = bijectionvector(K*T);
  
  
  for(unsigned int tt = 0; tt < chain_length; tt++){
    parm_update_rRUM(N,Jt,K,T,Alphas_init,pi_init,taus_init,R,r_stars_init,pi_stars_init,
                     Qs, Response, X, Smats_init, Gmats_init, test_order, Test_versions,
                     dirich_prior);
    
    if(tt>=burn_in){
      unsigned int tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
        r_stars.slice(tmburn).rows(Jt*t,(Jt*(t+1)-1)) = r_stars_init.slice(t);
        pi_stars.rows(Jt*t,(Jt*(t+1)-1)).col(tmburn) = pi_stars_init.col(t);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      taus.col(tmburn) = taus_init;
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("r_stars",r_stars),
                            Rcpp::Named("pi_stars",pi_stars),
                            Rcpp::Named("pis",pis),
                            Rcpp::Named("taus",taus));
}



//[[Rcpp::export]]
void parm_update_NIDA_indept(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                             arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, const arma::cube Qs, 
                             const arma::cube& responses, arma::cube& X_ijk, arma::cube& Smats, arma::cube& Gmats,
                             const arma::mat& test_order,const arma::vec& Test_versions, const arma::vec& dirich_prior){
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj,prodXijk,pi_ijk,aik,u,compare;
  double pi_ik,aik_nmrtr_k,aik_dnmntr_k,c_aik_1,c_aik_0,ptranspost_1,ptranspost_0,ptransprev_1,ptransprev_0;
  arma::vec aik_nmrtr(K);
  arma::vec aik_dnmntr(K);
  double D_bar = 0;
  arma::mat Classes(N,(T));
  
  // update X
  for(unsigned int i=1;i<N;i++){
    unsigned int test_version_it = Test_versions(i)-1;
    for(unsigned int t=0; t<(T); t++){
      unsigned int block = test_order(test_version_it,t)-1;
      arma::mat Q_it = Qs.slice(block);
      arma::vec alpha_i =(alphas.slice(t).row(i)).t();
      arma::vec Yi =(responses.slice(t).row(i)).t();
      arma::vec ui = arma::randu<arma::vec>(K);
      aik_nmrtr    = arma::ones<arma::vec>(K);
      aik_dnmntr   = arma::ones<arma::vec>(K);
      
      for(unsigned int j=0;j<Jt;j++){
        double Yij = Yi(j);
        // Note that the Xijk cube is indexed from j = 1 to Jk*(T+1)
        unsigned int j_star = block*Jt+j;
        arma::vec Xij = X_ijk.tube(i,j_star); 
        arma::uvec task_ij = find(Q_it.row(j) == 1);
        
        for(unsigned int k = 0;k<task_ij.n_elem ;k++){
          kj = task_ij(k);
          aik = alpha_i(kj);
          Xij(kj) = 1;
          prodXijk = prod(Xij(task_ij));
          u = R::runif(0.0,1.0);
          pi_ijk = (1.0-prodXijk)*(aik*(1.0-Smats(j,kj,block)) + (1.0-aik)*Gmats(j,kj,block) );
          compare=(pi_ijk>u);
          Xij(kj)=(1.0-Yij)*compare + Yij;
          
          aik_nmrtr(kj) = ( Xij(kj)*(1.0-Smats(j,kj,block)) + (1.0-Xij(kj))*Smats(j,kj,block) )*aik_nmrtr(kj);
          aik_dnmntr(kj) = ( Xij(kj)*Gmats(j,kj,block) + (1.0-Xij(kj))*(1.0-Gmats(j,kj,block)) )*aik_dnmntr(kj);
        }
        X_ijk.tube(i,j_star) = Xij;
      }
      
      //Update alpha_ikt
      for(unsigned int k=0;k<K;k++){
        arma::vec alpha_i_1 = alpha_i;
        alpha_i_1(k) = 1.0;
        c_aik_1 = (arma::conv_to< double >::from( alpha_i_1.t()*bijectionvector(K) ));
        arma::vec alpha_i_0 = alpha_i;
        alpha_i_0(k) = 0.0;
        c_aik_0 = (arma::conv_to< double >::from( alpha_i_0.t()*bijectionvector(K) ));
        // Rcpp::Rcout<<alpha_i_1<<std::endl;
        // Rcpp::Rcout<<alpha_i_0<<std::endl;
        
        // initial time point
        if(t == 0){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          aik_nmrtr_k = aik_nmrtr(k)*pi(c_aik_1)*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*pi(c_aik_0)*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        
        // middle points
        if(t > 0 && t < (T-1)){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        // last time point
        if(t == (T-1)){
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
      }
      alphas.slice(t).row(i) = alpha_i.t();
      
      // Get DIC
      arma::vec Svec = Smats.slice(0).row(0).t();
      arma::vec Gvec = Gmats.slice(0).row(0).t();
      D_bar += log(pYit_NIDA(alpha_i,Yi,Svec,Gvec,Q_it));
    }
  }
  
  for(unsigned int t = 0; t<(T); t++){
    Classes.col(t) = alphas.slice(t) * bijectionvector(K);
  }
  
  arma::vec CLASS_0 = alphas.slice(0) * bijectionvector(K);
  
  // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +dirich_prior;
  pi = rDirichlet(deltatilde);
  
  // update item parameters
  //update Smat and Gmat
  double pg,ps,ug,us,gk,sk,bsk,ask,bgk,agk, aikt,Xijk,qijk;
  
  for(unsigned int k = 0;k<K;k++){
    bsk = ask = agk = bgk = 0;
    for(unsigned int t = 0; t<(T); t++){
      for(unsigned int i = 0; i<N; i++){
        aikt = alphas(i,k,t);
        unsigned int test_version_it = Test_versions(i)-1;
        unsigned int block = test_order(test_version_it,t)-1;
        for(unsigned int j = 0; j<Jt; j++){
          unsigned int j_star = block*Jt+j;
          Xijk = X_ijk(i,j_star,k); 
          qijk = Qs(j,k,block);
          ask += aikt*(1-Xijk)*qijk;
          bsk += aikt*Xijk*qijk;
          agk += (1-aikt)*Xijk*qijk;
          bgk += (1-aikt)*(1-Xijk)*qijk;
        }
      }
    }
    ug = R::runif(0.0,1.0);
    us = R::runif(0.0,1.0);
    
    //draw g conditoned upon s_t-1
    pg = R::pbeta(1.0-Smats(0,k,0),agk+1.0,bgk+1.0,1,0);
    gk = R::qbeta(ug*pg,agk+1.0,bgk+1.0,1,0);
    //draw s conditoned upon g
    ps = R::pbeta(1.0-gk,ask+1.0,bsk+1.0,1,0);
    sk = R::qbeta(us*ps,ask+1.0,bsk+1.0,1,0);
    
    Gmats.tube(0,k,(Jt-1),k).fill(gk);
    Smats.tube(0,k,(Jt-1),k).fill(sk);
  }
  
  // Update transition probabilities
  for(unsigned int k = 0; k<K; k++){
    double a_tau, b_tau;
    a_tau = b_tau = 1;
    arma::uvec prereqs = arma::find(R.row(k)==1);
    if(prereqs.n_elem==0){
      for(unsigned int t =0; t<(T-1); t++){
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    if(prereqs.n_elem>0){
      for(unsigned int t=0; t<(T-1); t++){
        arma::mat a_prereqs = alphas.slice(t+1).cols(prereqs);
        arma::vec a_preq_1 = arma::prod(a_prereqs,1);
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0 && a_preq_1 == 1);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    taus(k) = R::rbeta((a_tau+1), (b_tau+1));
  }
}


//[[Rcpp::export]]
Rcpp::List Gibbs_NIDA_indept(const arma::cube& Response, const arma::cube& Qs, const arma::mat& R,
                             const arma::mat& test_order, const arma::vec& Test_versions,
                             const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  
  // initialize parameters
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec dirich_prior = arma::ones<arma::vec>(nClass);
  for(unsigned int cc = 0; cc < nClass; cc++){
    arma::vec alpha_cc = inv_bijectionvector(K,cc);
    for(unsigned int k = 0; k<K; k++){
      arma::uvec prereqs = arma::find(R.row(k)==1);
      if(prereqs.n_elem==0){
        if(alpha_cc(k)==1 && arma::prod(alpha_cc(prereqs))==0){
          pi_init(cc) = 0;
          dirich_prior(cc) = 0;
        }
      }
    }
  }
  pi_init = pi_init/arma::sum(pi_init);
  
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  for(unsigned int i = 0; i < N; i++){
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  
  arma::vec taus_init = .5*arma::ones<arma::vec>(K);
  
  arma::cube Alphas_init = simulate_alphas_indept(taus_init,Alphas_0_init,T,R);
  
  arma::cube r_stars_init = .5 + .2*arma::randu<arma::cube>(Jt,K,T);
  arma::mat pi_stars_init = .7 + .2*arma::randu<arma::mat>(Jt,T);
  
  arma::cube Smats_init = arma::randu<arma::cube>(Jt,K,T);
  arma::cube Gmats_init = arma::randu<arma::cube>(Jt,K,T) % (1-Smats_init);
  
  arma::cube X = arma::ones<arma::cube>(N,(Jt*T),K);
  
  
  // Create objects for storage 
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat Ss(K,(chain_length-burn_in));
  arma::mat Gs(K,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat taus(K,(chain_length-burn_in));
  
  arma::mat Trajectories_mat(N,(K*T));
  arma::vec vv = bijectionvector(K*T);
  
  for(unsigned int tt = 0; tt < chain_length; tt++){
    parm_update_NIDA_indept(N,Jt,K,T,Alphas_init,pi_init,taus_init,R,
                            Qs, Response, X, Smats_init, Gmats_init, test_order, Test_versions,
                            dirich_prior);
    
    if(tt>=burn_in){
      unsigned int tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
        Ss.col(tmburn) = Smats_init.slice(0).row(0).t();
        Gs.col(tmburn) = Gmats_init.slice(0).row(0).t();
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      taus.col(tmburn) = taus_init;
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",Ss),
                            Rcpp::Named("gs",Gs),
                            Rcpp::Named("pis",pis),
                            Rcpp::Named("taus",taus));
}





// [[Rcpp::export]]
void parm_update_DINA_FOHM(unsigned int N,unsigned int J,unsigned int K,unsigned int nClass,
                           unsigned int nT,const arma::cube& Y,const arma::mat& TP,
                           const arma::mat& ETA,arma::vec& ss,arma::vec& gs,arma::mat& CLASS,
                           arma::vec& pi,arma::mat& Omega){
  arma::vec pt_tm1(nClass);
  double cit,class_itp1,class_itm1,us,ug,pg,ps,gnew,snew,sold;
  arma::mat itempars(J,2);
  itempars.col(0) = ss;
  itempars.col(1) = gs;
  
  //update theta classes over times
  for(unsigned int i=0;i<N;i++){
    
    for(unsigned int t=0;t<nT;t++){
      //***select nonmissing y
      arma::rowvec Yit_temp = Y.subcube(i,0,t,i,J-1,t);
      arma::uvec nomiss_it = arma::find_finite(Yit_temp);
      arma::vec Yit = Yit_temp(nomiss_it);
      
      if(t==0){
        class_itp1 = CLASS(i,t+1);
        pt_tm1 = pi%Omega.col(class_itp1);
        arma::uvec pflag = find(TP.col(class_itp1)==1);
        arma::vec pY(pflag.n_elem);
        
        for(unsigned int g=0;g<pflag.n_elem;g++){
          double cc = pflag(g);
          //***select subset of rows for items
          arma::vec ETA_it_temp = ETA.col(cc);
          arma::vec ETA_it = ETA_it_temp(nomiss_it);
          //***need to select subelements of ss and gs
          pY(g) = pYit_DINA(ETA_it,Yit,itempars.rows(nomiss_it));
        }
        arma::vec numerator = pY % pt_tm1(pflag);
        arma::vec PS = numerator/arma::sum(numerator);
        cit = rmultinomial(PS);
        CLASS(i,t) = pflag(cit);
      }
      
      if(t==nT-1){
        class_itm1 = CLASS(i,t-1);
        pt_tm1 = (Omega.row(class_itm1)).t();
        arma::uvec pflag = find(TP.row(class_itm1)==1);
        arma::vec pY(pflag.n_elem);
        
        for(unsigned int g=0;g<pflag.n_elem;g++){
          double cc = pflag(g);
          //***select subset of rows for items
          arma::vec ETA_it_temp = ETA.col(cc);
          arma::vec ETA_it = ETA_it_temp(nomiss_it);
          //***need to select subelements of ss and gs
          pY(g) = pYit_DINA(ETA_it,Yit,itempars.rows(nomiss_it));
        }
        arma::vec numerator = pY % pt_tm1(pflag);
        arma::vec PS = numerator/arma::sum(numerator);
        cit = rmultinomial(PS);
        CLASS(i,t) = pflag(cit);
      }
      
      if( (t>0) & (t<nT-1) ){
        class_itm1 = CLASS(i,t-1);
        class_itp1 = CLASS(i,t+1);
        
        if(class_itm1==class_itp1 ){
          CLASS(i,t) = class_itm1;
        }
        if(class_itm1!=class_itp1 ){
          arma::vec c_temp = (TP.row(class_itm1)).t()%TP.col(class_itp1);
          arma::uvec pflag = find(c_temp==1);
          pt_tm1 = (Omega.row(class_itm1)).t()%Omega.col(class_itp1);
          arma::vec pY(pflag.n_elem);
          
          for(unsigned int g=0;g<pflag.n_elem;g++){
            double cc = pflag(g);
            //***select subset of rows for items
            arma::vec ETA_it_temp = ETA.col(cc);
            arma::vec ETA_it = ETA_it_temp(nomiss_it);
            //***need to select subelements of ss and gs
            pY(g) = pYit_DINA(ETA_it,Yit,itempars.rows(nomiss_it));
          }
          arma::vec numerator = pY % pt_tm1(pflag);
          arma::vec PS = numerator/arma::sum(numerator);
          cit = rmultinomial(PS);
          CLASS(i,t) = pflag(cit);
        }
      }
    }
  }
  //update pi
  arma::uvec class_sum=arma::hist(CLASS.col(0),arma::linspace<arma::vec>(0,nClass-1,nClass));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  //update Omega
  arma::mat tran_sum = arma::zeros<arma::mat>(nClass,nClass);
  for(unsigned int t=0;t<nT-1;t++){
    tran_sum = crosstab(CLASS.col(t),CLASS.col(t+1),TP,nClass,nClass) + tran_sum;
  }
  for(unsigned int cc=0;cc<nClass-1;cc++){
    arma::uvec class_ps = find(TP.row(cc)==1);
    arma::vec temp_mat = (tran_sum.row(cc)).t();
    arma::vec delta_tilde = temp_mat(class_ps) +1.;
    arma::vec w_c = rDirichlet(delta_tilde);
    
    for(unsigned int h=0;h<w_c.n_elem;h++){
      Omega(cc,class_ps(h)) = w_c(h);
    }
  }
  
  //update s,g
  arma::mat Ycrosstabmat = arma::ones<arma::mat>(nClass,2);
  for(unsigned int j=0;j<J;j++){
    arma::mat Yeta_sum = arma::zeros<arma::mat>(nClass,2);
    for(unsigned int t=0;t<nT;t++){
      //select nonmissing elements for each item
      arma::vec Yjt_temp = Y.subcube(0,j,t,N-1,j,t);
      arma::uvec nomiss_jt = arma::find_finite(Yjt_temp);
      arma::vec Yjt = Yjt_temp(nomiss_jt);
      //select classes for individuals with nonmissing yij
      arma::vec CLASS_t = CLASS.col(t);
      Yeta_sum = crosstab(CLASS_t(nomiss_jt),Yjt,Ycrosstabmat,nClass,2)+Yeta_sum;
    }
    arma::uvec ETAj1 = find(ETA.row(j)==1);
    arma::mat Yeta_sum_1 = Yeta_sum.rows(ETAj1);
    arma::rowvec ab_s = arma::sum(Yeta_sum_1);
    arma::uvec ETAj0 = find(ETA.row(j)==0);
    arma::mat Yeta_sum_0 = Yeta_sum.rows(ETAj0);
    arma::rowvec ab_g = arma::sum(Yeta_sum_0);
    
    //sample s and g as linearly truncated bivariate beta
    us=R::runif(0,1);
    ug=R::runif(0,1);
    sold = ss(j);
    //draw g conditoned upon s_t-1
    pg = R::pbeta(1.0-sold,ab_g(1)+1.,ab_g(0)+1.,1,0);
    gnew = R::qbeta(ug*pg,ab_g(1)+1.,ab_g(0)+1.,1,0);
    //draw s conditoned upon g
    ps = R::pbeta(1.0-gnew,ab_s(0)+1.,ab_s(1)+1.,1,0);
    snew = R::qbeta(us*ps,ab_s(0)+1.,ab_s(1)+1.,1,0);
    
    gs(j) = gnew;
    ss(j) = snew;
  }
}




// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_FOHM(const arma::cube& Y,const arma::mat& Q,
                           unsigned int burnin,unsigned int chain_length){
  unsigned int N = Y.n_rows;
  unsigned int J = Y.n_cols;
  unsigned int nT = Y.n_slices;
  unsigned int K = Q.n_cols;
  unsigned int C = pow(2,K);
  unsigned int chain_m_burn = chain_length-burnin;
  unsigned int tmburn;
  
  arma::vec vv = bijectionvector(K);
  arma::mat ETA = ETAmat(K,J,Q);
  arma::mat TP = TPmat(K);
  arma::vec vvp = bijectionvector(K*nT);
  
  //Savinging output
  arma::mat SS(J,chain_m_burn);
  arma::mat GS(J,chain_m_burn);
  arma::mat PIs(C,chain_m_burn);
  arma::cube OMEGAS(C,C,chain_m_burn);
  // arma::cube CLASStotal(N,nT,chain_m_burn);
  arma::mat Trajectories(N,(chain_m_burn));
  arma::mat Trajectories_mat(N,(K*nT));
  
  //need to initialize, alphas, X,ss, gs,pis 
  arma::mat Omega = rOmega(TP);
  arma::vec class0 = arma::randi<arma::vec>(N,arma::distr_param(0,C-1));
  arma::mat CLASS=rAlpha(Omega,N,nT,class0);
  arma::vec ss = arma::randu<arma::vec>(J);
  arma::vec gs = (arma::ones<arma::vec>(J) - ss)%arma::randu<arma::vec>(J);
  arma::vec delta0 = arma::ones<arma::vec>(C);
  arma::vec pis = rDirichlet(delta0);
  
  //Start Markov chain
  for(unsigned int t = 0; t < chain_length; t++){
    parm_update_DINA_FOHM(N,J,K,C,nT,Y,TP,ETA,ss,gs,CLASS,pis,Omega);
    
    if(t>=burnin){
      tmburn = t-burnin;
      //update parameter value via pointer. save classes and PIs
      SS.col(tmburn)       = ss;
      GS.col(tmburn)       = gs;
      PIs.col(tmburn)      = pis;
      OMEGAS.slice(tmburn) = Omega;
      for(unsigned int i = 0; i<N; i++){
        for(unsigned int tt = 0; tt < nT; tt++){
          Trajectories_mat.cols(K*tt,(K*(tt+1)-1)).row(i) = inv_bijectionvector(K,CLASS(i,tt)).t();
        }
      }
      Trajectories.col(tmburn) = Trajectories_mat * vvp;
    }
    
    if(t%1000==0){
      Rcpp::Rcout<<t<<std::endl;
    }
    
    
  }
  return Rcpp::List::create(Rcpp::Named("ss",SS),
                            Rcpp::Named("gs",GS),
                            Rcpp::Named("pis",PIs),
                            Rcpp::Named("omegas",OMEGAS),
                            Rcpp::Named("trajectories",Trajectories)
  );
}



//' @title Gibbs sampler for learning models
//' @description Runs MCMC to estimate parameters of any of the listed learning models. 
//' @param Response_list A \code{list} of dichotomous item responses. t-th element is an N-by-Jt matrix of responses at time t.
//' @param Q_list A \code{list} of Q-matrices. b-th element is a Jt-by-K Q-matrix for items in block b. 
//' @param model A \code{charactor} of the type of model fitted with the MCMC sampler, possible selections are 
//' "DINA_HO": Higher-Order Hidden Markov Diagnostic Classification Model with DINA responses;
//' "DINA_HO_RT_joint": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and joint modeling of latent
//' speed and learning ability; 
//' "DINA_HO_RT_sep": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and separate modeling of latent
//' speed and learning ability; 
//' "rRUM_indept": Simple independent transition probability model with rRUM responses
//' "NIDA_indept": Simple independent transition probability model with NIDA responses
//' "DINA_FOHM": First Order Hidden Markov model with DINA responses
//' @param test_order A \code{matrix} of the order of item blocks for each test version.
//' @param Test_versions A \code{vector} of the test version of each learner.
//' @param chain_length An \code{int} of the MCMC chain length.
//' @param burn_in An \code{int} of the MCMC burn-in chain length.
//' @param Q_examinee Optional. A \code{list} of the Q matrix for each learner. i-th element is a J-by-K Q-matrix for all items learner i was administered.
//' @param Latency_list Optional. A \code{list} of the response times. t-th element is an N-by-Jt matrix of response times at time t.
//' @param G_version Optional. An \code{int} of the type of covariate for increased fluency (1: G is dichotomous depending on whether all skills required for
//' current item are mastered; 2: G cumulates practice effect on previous items using mastered skills; 3: G is a time block effect invariant across 
//' subjects with different attribute trajectories)
//' @param theta_propose Optional. A \code{scalar} for the standard deviation of theta's proposal distribution in the MH sampling step.
//' @param deltas_propose Optional. A \code{vector} for the band widths of each lambda's proposal distribution in the MH sampling step.
//' @param R Optional. A reachability \code{matrix} for the hierarchical relationship between attributes. 
//' @return A \code{list} of parameter samples and Metropolis-Hastings acceptance rates (if applicable).
//' @author Susu Zhang
//' @examples
//' output_FOHM = MCMC_learning(Y_sim_list,Q_list,"DINA_FOHM",test_order,Test_versions,10000,5000)
//' @export
// [[Rcpp::export]]
Rcpp::List MCMC_learning(const Rcpp::List Response_list, const Rcpp::List Q_list, 
                          const String model, const arma::mat& test_order, const arma::vec& Test_versions,
                          const unsigned int chain_length, const unsigned int burn_in,
                          const Rcpp::Nullable<Rcpp::List> Q_examinee=R_NilValue,
                          const Rcpp::Nullable<Rcpp::List> Latency_list = R_NilValue, const int G_version = NA_INTEGER,
                          const double theta_propose = 0., const Rcpp::Nullable<Rcpp::NumericVector> deltas_propose = R_NilValue,
                          const Rcpp::Nullable<Rcpp::NumericMatrix> R = R_NilValue){
  Rcpp::List output;
  unsigned int T = test_order.n_rows;
  arma::mat temp = Rcpp::as<arma::mat>(Q_list[0]);
  unsigned int Jt = temp.n_rows;
  unsigned int K = temp.n_cols;
  unsigned int N = Test_versions.n_elem;
  arma::cube Response(N,Jt,T);
  arma::cube Latency(N,Jt,T);
  arma::cube Qs(Jt,K,T);
  for(unsigned int t = 0; t<T; t++){
    Response.slice(t) = Rcpp::as<arma::mat>(Response_list[t]);
    Qs.slice(t) = Rcpp::as<arma::mat>(Q_list[t]);
    if(Latency_list.isNotNull()){
      Rcpp::List tmp = Rcpp::as<Rcpp::List>(Latency_list);
      Latency.slice(t) = Rcpp::as<arma::mat>(tmp[t]);
    }
  }
  if(model == "DINA_HO"){
    
    output = Gibbs_DINA_HO(Response, Qs, Rcpp::as<Rcpp::List>(Q_examinee), test_order, Test_versions, theta_propose, Rcpp::as<arma::vec>(deltas_propose),
                           chain_length, burn_in);
  }
  if(model == "DINA_HO_RT_joint"){
    output = Gibbs_DINA_HO_RT_joint(Response, Latency, Qs, Rcpp::as<Rcpp::List>(Q_examinee), test_order, Test_versions, G_version,
                                    theta_propose, Rcpp::as<arma::vec>(deltas_propose), chain_length, burn_in);
  }
  if(model == "DINA_HO_RT_sep"){
    output = Gibbs_DINA_HO_RT_sep(Response, Latency, Qs, Rcpp::as<Rcpp::List>(Q_examinee), test_order, Test_versions, G_version,
                                  theta_propose, Rcpp::as<arma::vec>(deltas_propose), chain_length, burn_in);
  }
  if(model == "rRUM_indept"){
    output = Gibbs_rRUM_indept(Response, Qs, Rcpp::as<arma::mat>(R),test_order, Test_versions, chain_length, burn_in);
  }
  if(model == "NIDA_indept"){
    output = Gibbs_NIDA_indept(Response, Qs, Rcpp::as<arma::mat>(R), test_order, Test_versions, chain_length, burn_in);
  }
  if(model == "DINA_FOHM"){
    arma::cube Y_miss = resp_miss(Response, test_order, Test_versions);
    unsigned int Jt = Qs.n_rows;
    unsigned int T = Qs.n_slices;
    unsigned int K = Qs.n_cols;
    arma::mat Q_mat(Jt*T, K);
    for(unsigned int t= 0; t<T; t++){
      Q_mat.rows(Jt*t, (Jt*(t+1)-1)) = Qs.slice(t);
    }
    output = Gibbs_DINA_FOHM(Y_miss, Q_mat, burn_in, chain_length);
  }
  
  return(output);
}



// ------------------ Output Extraction ----------------------------------------------------------
//    Functions for computing point estimates, DIC, and posterior predictive probabilities      
// -----------------------------------------------------------------------------------------------




//' @title Obtain learning model point estimates
//' @description Obtain EAPs of continuous parameters and EAP or MAP of the attribute trajectory estimates under
//' the CDM learning models based on the MCMC output
//' @param output A \code{list} of MCMC outputs, obtained from the MCMC_learning function
//' @param model A \code{charactor} of the type of model fitted with the MCMC sampler, possible selections are 
//' "DINA_HO": Higher-Order Hidden Markov Diagnostic Classification Model with DINA responses;
//' "DINA_HO_RT_joint": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and joint modeling of latent
//' speed and learning ability; 
//' "DINA_HO_RT_sep": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and separate modeling of latent
//' speed and learning ability; 
//' "rRUM_indept": Simple independent transition probability model with rRUM responses
//' "NIDA_indept": Simple independent transition probability model with NIDA responses
//' "DINA_FOHM": First Order Hidden Markov model with DINA responses
//' @param N An \code{int} of number of subjects 
//' @param Jt An \code{int} of number of items in each block
//' @param K An \code{int} of number of skills
//' @param T An \code{int} of number of time points
//' @param alpha_EAP A \code{boolean} operator (T/F) of whether to use EAP for alphas (if F: use most likely trajectory (MAP) for alphas) 
//' @return A \code{list} of point estimates of model parameters
//' @author Susu Zhang
//' @examples
//' point_estimates = point_estimates_learning(output_FOHM,"DINA_FOHM",N,Jt,K,T,alpha_EAP = T)
//' @export
// [[Rcpp::export]]
Rcpp::List point_estimates_learning(const Rcpp::List output, const String model, const unsigned int N,
                                    const unsigned int Jt, const unsigned int K, const unsigned int T,
                                    bool alpha_EAP = true){
  Rcpp::List point_ests;
  // extract common outputs
  arma::mat Traject = Rcpp::as<arma::mat>(output["trajectories"]);
  arma::mat pis = Rcpp::as<arma::mat>(output["pis"]);
  unsigned int n_its = Traject.n_cols;
  
  // compute Alpha_hat
  arma::cube Alphas_est = arma::zeros<arma::cube>(N,K,T);
  arma::mat Alphas_i_mat(n_its,K*T);
  if(alpha_EAP==true){                                   // Compute EAP for alphas
    for(unsigned int i = 0; i<N; i++){
      for(unsigned int tt = 0; tt<n_its; tt++){
        Alphas_i_mat.row(tt) = inv_bijectionvector((K*T),Traject(i,tt)).t();
      }
      for(unsigned int kk = 0; kk<(K*T); kk++){
        if(arma::mean(Alphas_i_mat.col(kk))>.5){
          int k_star = kk % K;
          int t_star = (kk-k_star)/K;
          Alphas_est(i,k_star,t_star) = 1;
        }
      }
    }
  }else{                                                // Find most likely trajectory
    for(unsigned int i= 0; i<N; i++){
      arma::vec traject_sorted = arma::sort(Traject.row(i).t());
      double traject_ML = getMode(traject_sorted,n_its);
      arma::vec alpha_i = inv_bijectionvector((K*T),traject_ML);
      for(unsigned int t = 0; t<T; t++){
        Alphas_est.slice(t).row(i) = alpha_i.subvec(K*t, (K*(t+1)-1)).t();
      }
    }
  }
  arma::vec pis_EAP = arma::mean(pis,1);
  
  //"DINA_HO", "DINA_HO_RT_joint", "DINA_HO_RT_sep", "rRUM_indept","NIDA_indept","DINA_FOHM"
  if(model == "DINA_HO"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("thetas_EAP",thetas_EAP),
                                    Rcpp::Named("lambdas_EAP",lambdas_EAP));
  }
  
  if(model == "DINA_HO_RT_sep"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    
    arma::mat tauvar = Rcpp::as<arma::mat>(output["tauvar"]);
    double tauvar_EAP = arma::mean(tauvar.col(0));
    
    
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("as_EAP",as_EAP),
                                    Rcpp::Named("gammas_EAP",gammas_EAP),
                                    Rcpp::Named("thetas_EAP",thetas_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP),
                                    Rcpp::Named("lambdas_EAP",lambdas_EAP),
                                    Rcpp::Named("phis",phi_EAP),
                                    Rcpp::Named("tauvar_EAP",tauvar_EAP));
    
  }
  
  if(model == "DINA_HO_RT_joint"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    
    arma::cube Sigs = Rcpp::as<arma::cube>(output["Sigs"]);
    arma::mat Sigs_EAP = arma::mean(Sigs, 2);
    
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("as_EAP",as_EAP),
                                    Rcpp::Named("gammas_EAP",gammas_EAP),
                                    Rcpp::Named("thetas_EAP",thetas_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP),
                                    Rcpp::Named("lambdas_EAP",lambdas_EAP),
                                    Rcpp::Named("phis",phi_EAP),
                                    Rcpp::Named("Sigs_EAP",Sigs_EAP));
    
  }
  
  if(model == "rRUM_indept"){
    arma::cube r_stars = Rcpp::as<arma::cube>(output["r_stars"]);
    arma::mat r_stars_EAP = arma::mean(r_stars,2);
    
    arma::mat pi_stars = Rcpp::as<arma::mat>(output["pi_stars"]);
    arma::vec pi_stars_EAP = arma::mean(pi_stars,1);
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("r_stars_EAP",r_stars_EAP),
                                    Rcpp::Named("pi_stars_EAP",pi_stars_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP));
    
  }
  
  if(model == "NIDA_indept"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP));
    
  }
  
  if(model == "DINA_FOHM"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::cube omegas = Rcpp::as<arma::cube>(output["omegas"]);
    arma::mat omegas_EAP = arma::mean(omegas,2);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("omegas_EAP",omegas_EAP));
    
  }
  
  
  return(point_ests);
}

//' @title Model fit statistics of learning models
//' @description Obtain joint model's deviance information criteria (DIC) and posterior predictive item means, item response time means, 
//' item odds ratios, subject total scores at each time point, and subject total response times at each time point.
//' @param output A \code{list} of MCMC outputs, obtained from the MCMC_learning function
//' @param model A \code{charactor} of the type of model fitted with the MCMC sampler, possible selections are 
//' "DINA_HO": Higher-Order Hidden Markov Diagnostic Classification Model with DINA responses;
//' "DINA_HO_RT_joint": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and joint modeling of latent
//' speed and learning ability; 
//' "DINA_HO_RT_sep": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and separate modeling of latent
//' speed and learning ability; 
//' "rRUM_indept": Simple independent transition probability model with rRUM responses
//' "NIDA_indept": Simple independent transition probability model with NIDA responses
//' "DINA_FOHM": First Order Hidden Markov model with DINA responses
//' @param Response_list A \code{list} of dichotomous item responses. t-th element is an N-by-Jt matrix of responses at time t.
//' @param Q_list A \code{list} of Q-matrices. b-th element is a Jt-by-K Q-matrix for items in block b. 
//' @param test_order A \code{matrix} of the order of item blocks for each test version.
//' @param Test_versions A \code{vector} of the test version of each learner.
//' @param Q_examinee Optional. A \code{list} of the Q matrix for each learner. i-th element is a J-by-K Q-matrix for all items learner i was administered.
//' @param Latency_list Optional. A \code{list} of the response times. t-th element is an N-by-Jt matrix of response times at time t.
//' @param G_version Optional. An \code{int} of the type of covariate for increased fluency (1: G is dichotomous depending on whether all skills required for
//' current item are mastered; 2: G cumulates practice effect on previous items using mastered skills; 3: G is a time block effect invariant across 
//' subjects with different attribute trajectories)
//' @param R Optional. A reachability \code{matrix} for the hierarchical relationship between attributes. 
//' @return A list of DIC matrix, with deviance decomposed to that of the transition model, response model, response time model (if applicable),
//' and joint model of random parameters, and posterior predictive item means, item odds ratios, item averaged response times, subjects' total
//' scores at each time point, and subjects' total response times at each time point. Predicted values can be compared to the observed ones from
//' empirical data.
//' @examples
//' FOHM_fit <- Learning_fit(output_FOHM,"DINA_FOHM",Y_sim_list,Q_list,test_order,Test_versions)
//' @export
// [[Rcpp::export]]
Rcpp::List Learning_fit(const Rcpp::List output, const String model,
                        const Rcpp::List Response_list, const Rcpp::List Q_list,
                        const arma::mat test_order, const arma::vec Test_versions,
                        const Rcpp::Nullable<Rcpp::List> Q_examinee=R_NilValue,
                        const Rcpp::Nullable<Rcpp::List> Latency_list = R_NilValue, const int G_version = NA_INTEGER,
                        const Rcpp::Nullable<Rcpp::NumericMatrix> R = R_NilValue){
  
  unsigned int T = test_order.n_rows;
  arma::mat temp = Rcpp::as<arma::mat>(Q_list[0]);
  unsigned int Jt = temp.n_rows;
  unsigned int K = temp.n_cols;
  unsigned int N = Test_versions.n_elem;
  arma::cube Response(N,Jt,T);
  arma::cube Latency(N,Jt,T);
  arma::cube Qs(Jt,K,T);
  for(unsigned int t = 0; t<T; t++){
    Response.slice(t) = Rcpp::as<arma::mat>(Response_list[t]);
    Qs.slice(t) = Rcpp::as<arma::mat>(Q_list[t]);
    if(Latency_list.isNotNull()){
      Rcpp::List tmp = Rcpp::as<Rcpp::List>(Latency_list);
      Latency.slice(t) = Rcpp::as<arma::mat>(tmp[t]);
    }
  }
  arma::mat Traject = Rcpp::as<arma::mat>(output["trajectories"]);
  arma::mat pis = Rcpp::as<arma::mat>(output["pis"]);
  arma::vec pis_EAP = arma::mean(pis,1);
  unsigned int n_its = Traject.n_cols;
  arma::cube Alphas_est = arma::zeros<arma::cube>(N,K,T);
  arma::mat Alphas_i_mat(n_its,K*T);
  for(unsigned int i= 0; i<N; i++){
    arma::vec traject_sorted = arma::sort(Traject.row(i).t());
    double traject_ML = getMode(traject_sorted,n_its);
    arma::vec alpha_i = inv_bijectionvector((K*T),traject_ML);
    for(unsigned int t = 0; t<T; t++){
      Alphas_est.slice(t).row(i) = alpha_i.subvec(K*t, (K*(t+1)-1)).t();
    }
  }
  Rcpp::NumericMatrix DIC(3,5); 
  Rcpp::List posterior_predictives;
  
  arma::vec d_tran(n_its);
  arma::vec d_time(n_its);
  arma::vec d_response(n_its);
  arma::vec d_joint(n_its);
  arma::cube alphas(N,K,T);
  arma::vec G_it(Jt);
  arma::cube total_time_PP(N,T,n_its);
  arma::cube total_score_PP(N,T,n_its);
  arma::mat item_mean_PP(Jt*T, n_its);
  arma::cube item_OR_PP(Jt*T, Jt*T, n_its);
  arma::mat RT_mean_PP(Jt*T,n_its);
  
  arma::vec vv = bijectionvector(K);
  arma::cube ETA(Jt, (pow(2,K)), T);
  for(unsigned int t = 0; t<T; t++){
    ETA.slice(t) = ETAmat(K,Jt, Qs.slice(t));
  }
  
  arma::mat Y_sim_collapsed(N,Jt*T);
  arma::mat L_sim_collapsed(N,Jt*T);
  
  if(model == "DINA_HO"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // next get itempars and simulated responses
      
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      
      arma::cube Y_sim = simDINA(alphas,itempars_cube,ETA,test_order,Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas.col(tt), thetas(i,tt),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
          }
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_HO_sep(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        lambdas_EAP, thetas_EAP(i),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
        }
        // The log likelihood from response time model
        int test_block_it = test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
  }
  
  if(model == "DINA_HO_RT_sep"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    arma::mat tauvar = Rcpp::as<arma::mat>(output["tauvar"]);
    double tauvar_EAP = arma::mean(tauvar.col(0));
    arma::cube J_incidence = J_incidence_cube(test_order, Qs);
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // put item parameters into a matrix
      // next get itempars and simulated responses
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      arma::mat RT_itempars(Jt*T,2);
      RT_itempars.col(0) = as.col(tt);
      RT_itempars.col(1) = gammas.col(tt);
      arma::cube RT_itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
        RT_itempars_cube.slice(t) = RT_itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      arma::cube Y_sim = simDINA(alphas,itempars_cube,ETA,test_order,Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      arma::cube L_sim = sim_RT(alphas, RT_itempars_cube,Qs,taus.col(tt),phis(tt,0),
                                ETA, G_version, test_order, Test_versions);
      arma::mat L_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          L_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = L_sim.slice(t).row(i);
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas.col(tt), thetas(i,tt),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
          }
          if (G_version == 1) {
            G_it = ETA.slice(test_block_it).col(class_it);
          }
          if (G_version == 2) {
            G_it = G2vec_efficient(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                                   test_order, t);
          }
          if(G_version==3){
            G_it = arma::ones<arma::vec>(Jt);
            arma::vec y(Jt);y.fill((t+1.)/T);
            G_it =G_it % y;
          }
          // The loglikelihood from log-Normal RT model
          time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                                RT_itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                taus(i,tt), phis(tt,0)));
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          total_time_PP(i,t,tt) = arma::sum(L_sim.slice(t).row(i));
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)) + R::dnorm(taus(i,tt),0,std::sqrt(tauvar(tt,0)),true); 
      }
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      RT_mean_PP.col(tt) = arma::mean(L_sim_collapsed,0).t();
      

    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0) + DIC(0,1) + DIC(0,2) + DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    arma::mat RT_itempars_EAP(Jt*T,2);
    RT_itempars_EAP.col(0) = as_EAP;
    RT_itempars_EAP.col(1) = gammas_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_HO_sep(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        lambdas_EAP, thetas_EAP(i),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
        }
        // The log likelihood from response time model
        int test_block_it = test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from log-Normal RT model
        time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                              RT_itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                              taus_EAP(i),phi_EAP));
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0))  + R::dnorm(taus_EAP(i),0,std::sqrt(tauvar_EAP),true);
    }
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,1)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("RT_mean_PP",RT_mean_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP),
                                               Rcpp::Named("total_time_PP",total_time_PP));
    
    
  }
  
  if(model == "DINA_HO_RT_joint"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    arma::cube Sigs = Rcpp::as<arma::cube>(output["Sigs"]);
    arma::mat Sigs_EAP = arma::mean(Sigs, 2);
    arma::cube J_incidence = J_incidence_cube(test_order, Qs);
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // put item parameters into a matrix
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      arma::mat RT_itempars(Jt*T,2);
      RT_itempars.col(0) = as.col(tt);
      RT_itempars.col(1) = gammas.col(tt);
      arma::cube RT_itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
        RT_itempars_cube.slice(t) = RT_itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      arma::cube Y_sim = simDINA(alphas,itempars_cube,ETA,test_order,Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      arma::cube L_sim = sim_RT(alphas, RT_itempars_cube,Qs,taus.col(tt),phis(tt,0),
                                ETA, G_version, test_order, Test_versions);
      arma::mat L_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          L_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = L_sim.slice(t).row(i);
          
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_HO_joint(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            lambdas.col(tt), thetas(i,tt),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
          }
          if (G_version == 1) {
            G_it = ETA.slice(test_block_it).col(class_it);
          }
          if (G_version == 2) {
            G_it = G2vec_efficient(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                                   test_order, t);
          }
          if(G_version==3){
            G_it = arma::ones<arma::vec>(Jt);
            arma::vec y(Jt);y.fill((t+1.)/T);
            G_it =G_it % y;
          }
          // The loglikelihood from log-Normal RT model
          time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                                RT_itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                taus(i,tt), phis(tt,0)));
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          total_time_PP(i,t,tt) = arma::sum(L_sim.slice(t).row(i));
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        arma::vec thetatau(2);
        thetatau(0) = thetas(i,tt);
        thetatau(1) = taus(i,tt);
        joint += std::log(pis(class_i0,tt)) + std::log(dmvnrm(thetatau,arma::zeros<arma::vec>(2),Sigs.slice(tt))); 
      }
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      RT_mean_PP.col(tt) = arma::mean(L_sim_collapsed,0).t();
      

    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0) + DIC(0,1) + DIC(0,2) + DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    arma::mat RT_itempars_EAP(Jt*T,2);
    RT_itempars_EAP.col(0) = as_EAP;
    RT_itempars_EAP.col(1) = gammas_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_HO_joint(Alphas_est.slice(t).row(i).t(),
                                          Alphas_est.slice(t+1).row(i).t(),
                                          lambdas_EAP, thetas_EAP(i),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
        }
        // The log likelihood from response time model
        int test_block_it = test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from log-Normal RT model
        time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                              RT_itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                              taus_EAP(i),phi_EAP));
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      arma::vec thetatau(2);
      thetatau(0) = thetas_EAP(i);
      thetatau(1) = taus_EAP(i);
      joint += std::log(pis_EAP(class_i0))  + + std::log(dmvnrm(thetatau,arma::zeros<arma::vec>(2),Sigs_EAP));
    }
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,1)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
    
  }
  
  if(model == "rRUM_indept"){
    arma::cube r_stars = Rcpp::as<arma::cube>(output["r_stars"]);
    arma::mat r_stars_EAP = arma::mean(r_stars,2);
    arma::mat pi_stars = Rcpp::as<arma::mat>(output["pi_stars"]);
    arma::vec pi_stars_EAP = arma::mean(pi_stars,1);
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      arma::cube r_stars_cube(Jt,K,T);
      arma::mat pi_stars_mat(Jt,T);
      for(unsigned int t= 0; t<T; t++){
        r_stars_cube.slice(t) = r_stars.slice(tt).rows(Jt*t,(Jt*(t+1)-1));
        pi_stars_mat.col(t) = pi_stars.col(tt).subvec(Jt*t,(Jt*(t+1)-1));
      }
      arma::cube Y_sim = simrRUM(alphas, r_stars_cube, pi_stars_mat,Qs,test_order,Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = test_order(test_version_i,t)-1;
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_indept(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          taus.col(tt),Rcpp::as<arma::mat>(R)));
          }
          // The loglikelihood from the DINA
          response += std::log(pYit_rRUM(alphas.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                         pi_stars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)).col(tt),
                                         r_stars.slice(tt).rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                         Qs.slice(test_block_it)));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_indept(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        taus_EAP,Rcpp::as<arma::mat>(R)));
        }
        // The log likelihood from response time model
        int test_block_it = test_order(test_version_i, t) - 1;
        // The loglikelihood from the DINA
        response += std::log(pYit_rRUM(Alphas_est.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                       pi_stars_EAP.subvec((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                       r_stars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                       Qs.slice(test_block_it)));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
    
    
  }
  
  if(model == "NIDA_indept"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    
    
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      
      arma::cube Y_sim = simNIDA(alphas,ss.col(tt),gs.col(tt),Qs,test_order,Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = test_order(test_version_i,t)-1;
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_indept(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          taus.col(tt),Rcpp::as<arma::mat>(R)));
          }
          
          // The loglikelihood from the DINA
          response += std::log(pYit_NIDA(alphas.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                         ss.col(tt),
                                         gs.col(tt),
                                         Qs.slice(test_block_it)));
          
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);

    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_indept(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        taus_EAP,Rcpp::as<arma::mat>(R)));
        }
        // The log likelihood from response time model
        int test_block_it = test_order(test_version_i, t) - 1;
        // The loglikelihood from the DINA
        response += std::log(pYit_NIDA(Alphas_est.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                       ss_EAP,
                                       gs_EAP,
                                       Qs.slice(test_block_it)));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
    
  }
  
  if(model == "DINA_FOHM"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::cube omegas = Rcpp::as<arma::cube>(output["omegas"]);
    arma::mat omegas_EAP = arma::mean(omegas,2);
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // put item parameters into a matrix
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      arma::cube Y_sim = simDINA(alphas,itempars_cube,ETA,test_order,Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          
          // The transition model part
          if (t < (T - 1)) {
            int class_pre, class_post;
            class_pre = arma::dot(alphas.slice(t).row(i),vv);
            class_post = arma::dot(alphas.slice(t+1).row(i),vv);
            tran += std::log(omegas(class_pre,class_post,tt));
          }
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          int class_pre, class_post;
          class_pre = arma::dot(Alphas_est.slice(t).row(i),vv);
          class_post = arma::dot(Alphas_est.slice(t+1).row(i),vv);
          tran += std::log(omegas_EAP(class_pre,class_post));
        }
        // The log likelihood from response time model
        int test_block_it = test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
    
  }
  // get last row --- DIC 
  DIC.row(2) = 2. * DIC.row(0) - DIC.row(1);
  
  Rcpp::List dimnms = Rcpp::List::create(Rcpp::CharacterVector::create("D_bar","D(theta_bar)","DIC"),
                                                                       Rcpp::CharacterVector::create("Transition", "Response_Time","Response","Joint","Total"));
  
  DIC.attr("dimnames") = dimnms;
  
  return Rcpp::List::create(Rcpp::Named("DIC",DIC),
                            Rcpp::Named("PPs",posterior_predictives));
  
  
}


