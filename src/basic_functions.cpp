#include <RcppArmadillo.h>
#include "basic_functions.h"


// ------------------------------------------- Basic Functions ---------------------------------------------------
// Generating random numbers or generic functions for CDM (e.g., Q matrix, ETA, bijection, etc.)
// ---------------------------------------------------------------------------------------------------------------



// [[Rcpp::export]]
arma::vec bijectionvector(unsigned int K) {
  arma::vec vv(K);
  for(unsigned int k=0;k<K;k++){
    vv(k) = pow(2,K-k-1);
  }
  return vv;
}


//' @title Convert integer to attribute pattern
//' @description Based on the bijective relationship between natural numbers and sum of powers of two,
//'  convert integer between 0 and 2^K-1 to K-dimensional attribute pattern.
//' @param K An \code{int} for the number of attributes
//' @param CL An \code{int} between 0 and 2^K-1
//' @return A \code{vec} of the K-dimensional attribute pattern corresponding to CL.
//' @examples
//' inv_bijectionvector(4,0)
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
//' @param K An \code{int} of the number of attributes.
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
arma::cube resp_miss(const arma::cube& Responses, const arma::mat& Test_order, 
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
      unsigned int test_block_it = Test_order(Test_version_i,t)-1;
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
//' \donttest{
//' N = length(Test_versions)
//' J = nrow(Q_matrix)
//' K = ncol(Q_matrix)
//' L = nrow(Test_order)
//' Jt = J/L
//' OddsRatio(N,J,Y_real_array[,,1])}
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

// [[Rcpp::export]]
arma::cube Sparse2Dense(const arma::cube Y_real_array,
                        const arma::mat& Test_order,
                        const arma::vec& Test_versions){
  unsigned int N = Test_versions.n_elem;
  unsigned int J = Y_real_array.n_cols;
  unsigned int T = Y_real_array.n_slices;
  unsigned int Jt = J / T;

  arma::cube Response(N,Jt,T);
  for(unsigned int i=0; i<N; i++){
    unsigned int test_version = Test_versions(i);
    for(unsigned int t=0; t<T; t++){
      unsigned int test_index = Test_order(test_version-1,t);
      for(unsigned int j=0; j<Jt; j++){
        unsigned int item_index = (test_index-1)*Jt+j;
        Response(i,j,t) = Y_real_array(i,item_index,t);
      }
    }
  }
  return Response;
}

// [[Rcpp::export]]
arma::cube Dense2Sparse(const arma::cube Y_sim,
                        const arma::mat& Test_order,
                        const arma::vec& Test_versions){
  unsigned int N = Test_versions.n_elem;
  unsigned int Jt = Y_sim.n_cols;
  unsigned int T = Y_sim.n_slices;
  unsigned int J = Jt*T;
  
  arma::cube Y_sim_sparse(N,J,T);
  for(unsigned int i=0; i<N; i++){
    unsigned int test_version = Test_versions(i);
    for(unsigned int t=0; t<T; t++){
      unsigned int test_index = Test_order(test_version-1,t);
      for(unsigned int j=0; j<Jt; j++){
        unsigned int item_index = (test_index-1)*Jt+j;
        Y_sim_sparse(i,item_index,t) = Y_sim(i,j,t);
      }
    }
  }
  return Y_sim_sparse;
}

// [[Rcpp::export]]
arma::cube Mat2Array(const arma::mat Q_matrix, unsigned int T){
  unsigned int J = Q_matrix.n_rows;
  unsigned int K = Q_matrix.n_cols;  
  unsigned int Jt = J / T;
  
  arma::cube Q_array(Jt,K,T);
  for(unsigned int t=0; t<T; t++){
    for(unsigned int j=0; j<Jt; j++){
      for(unsigned int k=0; k<K; k++){
        unsigned int item_index = t*Jt+j;
        Q_array(j,k,t) = Q_matrix(item_index, k);
      }
    }
  }
  return Q_array;
}


// [[Rcpp::export]]
arma::mat Array2Mat(const arma::cube r_stars){
  unsigned int Jt = r_stars.n_rows;
  unsigned int T = r_stars.n_slices;
  unsigned int K = r_stars.n_cols;
  unsigned int J = Jt*T;
  arma::mat r_stars_mat(J,K);
  for(unsigned int j=0;j<Jt;j++){
    for(unsigned int t=0;t<T;t++){
      for(unsigned int k=0;k<K;k++){
        r_stars_mat(Jt*t+j,k) = r_stars(j,k,t);
      }
    }
  }
  return r_stars_mat;
}


//' @title Generate a list of Q-matrices for each examinee.
//' @description Generate a list of length N. Each element of the list is a JxK Q_matrix of all items
//' administered across all time points to the examinee, in the order of administration.
//' @param Q_matrix A J-by-K matrix, indicating the item-skill relationship.
//' @param Test_order A TxT matrix, each row is the order of item blocks for that test version.
//' @param Test_versions A vector of length N, containing each subject's test version.
//' @return A list of length N. Each element of the list is a JxK matrix.
//' @examples 
//' \donttest{
//' Q_examinee = Q_list(Q_matrix, Test_order, Test_versions)}
//' @export
// [[Rcpp::export]]
Rcpp::List Q_list(const arma::mat Q_matrix, const arma::mat Test_order, const arma::vec Test_versions){
  unsigned int N = Test_versions.n_elem;
  unsigned int J = Q_matrix.n_rows;
  unsigned int K = Q_matrix.n_cols;
  unsigned int T = Test_order.n_cols;
  unsigned int Jt = J/T;
  
  Rcpp::List Q_examinee(N);
  for(unsigned int i=0; i<N; i++){
    arma::mat Q_mat(J, K);
    arma::vec Test_order_i = Test_order.col(Test_versions(i)-1);
    for(unsigned int t=0; t<T; t++){
      Q_mat.submat(t*Jt,0,(t+1)*Jt-1,K-1) = Q_matrix.submat((Test_order_i(t)-1)*Jt,0,Test_order_i(t)*Jt-1,K-1);
    }
    Q_examinee[i] = Q_mat;
  }
  return Q_examinee;
}

