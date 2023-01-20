#ifndef TRANS_FUNCTIONS_H
#define TRANS_FUNCTIONS_H


arma::cube simulate_alphas_HO_sep_g(const arma::vec& lambdas, const arma::vec& thetas,
                                    const arma::mat& Q_matrix, const arma::cube& Design_array, 
                                    const arma::mat alpha0);


double pTran_HO_sep_g(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                      const arma::mat& Q_i, const arma::cube& Design_array, unsigned int t, unsigned int i);

arma::cube simulate_alphas_HO_joint_g(const arma::vec& lambdas, const arma::vec& thetas,
                                      const arma::mat& Q_matrix, const arma::cube& Design_array, 
                                      const arma::mat& alpha0);

double pTran_HO_joint_g(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                        const arma::mat& Q_i, const arma::cube Design_array, unsigned int t, unsigned int i);

arma::cube simulate_alphas_indept(const arma::vec taus, const arma::mat& alpha0s, const unsigned int L, const arma::mat& R);

arma::cube simulate_alphas_indept_g(const arma::vec taus, const unsigned int N, const unsigned int L, const arma::mat& R,
                                    const arma::mat alpha0);

double pTran_indept(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& taus,const arma::mat& R);

arma::cube simulate_alphas_FOHM(const arma::mat& Omega,unsigned int N,unsigned int L,
                                const arma::mat alpha0);
  
arma::mat rAlpha(const arma::mat& Omega,unsigned int N,unsigned int L, const arma::vec& alpha1);

arma::mat rOmega(const arma::mat& TP);  

arma::cube sim_alphas(const std::string model,
                      const Rcpp::Nullable<arma::vec&> lambdas, const Rcpp::Nullable<arma::vec&> thetas,
                      const Rcpp::Nullable<arma::mat&> Q_matrix, const Rcpp::Nullable<arma::cube&> Design_array, 
                      const Rcpp::Nullable<arma::vec> taus, const Rcpp::Nullable<arma::mat&> Omega,
                      int N, const int L, 
                      const Rcpp::Nullable<arma::mat&> R,
                      const Rcpp::Nullable<arma::mat> alpha0);  

#endif