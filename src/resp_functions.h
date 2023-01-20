#ifndef RESP_FUNCTIONS_H
#define RESP_FUNCTIONS_H

arma::vec sim_resp_DINA(unsigned int J, unsigned int K, const arma::mat& ETA, arma::vec& Svec, arma::vec& Gvec, arma::vec& alpha);


arma::cube simDINA_g(const arma::cube& alphas, const arma::mat& itempars, const arma::mat& Q_matrix,
                     const arma::cube& Design_array);
  
double pYit_DINA(const arma::vec& ETA_it,const arma::vec& Y_it, const arma::mat& itempars);

arma::vec sim_resp_rRUM(unsigned int J, unsigned int K, const arma::mat& Q,const arma::mat& rstar, const arma::vec& pistar,
                        const arma::vec& alpha);

arma::cube simrRUM_g(const arma::cube& alphas, const arma::mat& r_stars_mat, const arma::vec& pi_stars, 
                     const arma::mat Q_matrix, const arma::cube& Design_array);

double pYit_rRUM(const arma::vec& alpha_it, const arma::vec& Y_it, const arma::vec& pi_star_it, 
                 const arma::mat& r_star_it, const arma::mat& Q_it);

arma::vec sim_resp_NIDA(const unsigned int J, const unsigned int K, const arma::mat& Q, const arma::vec& Svec, const arma::vec& Gvec,
                        const arma::vec& alpha);

arma::cube simNIDA_g(const arma::cube& alphas, const arma::vec& Svec, const arma::vec& Gvec, 
                     const arma::mat Q_matrix, const arma::cube& Design_array);
  
double pYit_NIDA(const arma::vec& alpha_it, const arma::vec& Y_it, const arma::vec& Svec, 
                 const arma::vec& Gvec, const arma::mat& Q_it);

arma::cube sim_hmcdm(const std::string model,const arma::cube& alphas,const arma::mat& Q_matrix,const arma::cube& Design_array,
                     const Rcpp::Nullable<arma::mat> itempars,
                     const Rcpp::Nullable<arma::mat> r_stars,
                     const Rcpp::Nullable<arma::vec> pi_stars,
                     const Rcpp::Nullable<arma::vec> Svec,
                     const Rcpp::Nullable<arma::vec> Gvec
);
  
#endif