#ifndef RESP_FUNCTIONS_H
#define RESP_FUNCTIONS_H

arma::vec sim_resp_DINA(unsigned int J, unsigned int K, const arma::mat& ETA, arma::vec& Svec, arma::vec& Gvec, arma::vec& alpha);

arma::cube simDINA(const arma::cube& alphas, const arma::cube& itempars, const arma::cube& ETA,
                   const arma::mat& test_order, const arma::vec& Test_versions);

double pYit_DINA(const arma::vec& ETA_it,const arma::vec& Y_it, const arma::mat& itempars);

arma::vec sim_resp_rRUM(unsigned int J, unsigned int K, const arma::mat& Q,const arma::mat& rstar, const arma::vec& pistar,
                        const arma::vec& alpha);

arma::cube simrRUM(const arma::cube& alphas, const arma::cube& r_stars, const arma::mat& pi_stars, 
                   const arma::cube Qs, const arma::mat& test_order, const arma::vec& Test_versions);
                   
double pYit_rRUM(const arma::vec& alpha_it, const arma::vec& Y_it, const arma::vec& pi_star_it, 
                 const arma::mat& r_star_it, const arma::mat& Q_it);

arma::vec sim_resp_NIDA(const unsigned int J, const unsigned int K, const arma::mat& Q, const arma::vec& Svec, const arma::vec& Gvec,
                        const arma::vec& alpha);

arma::cube simNIDA(const arma::cube& alphas, const arma::vec& Svec, const arma::vec& Gvec, 
                   const arma::cube Qs, const arma::mat& test_order, const arma::vec& Test_versions);
                   
double pYit_NIDA(const arma::vec& alpha_it, const arma::vec& Y_it, const arma::vec& Svec, 
                 const arma::vec& Gvec, const arma::mat& Q_it);

#endif