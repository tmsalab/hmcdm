#ifndef MCMC_FUNCTIONS_H
#define MCMC_FUNCTIONS_H

Rcpp::List parm_update_HO(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                          arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                          const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                          const arma::mat test_order, const arma::vec Test_versions, 
                          const double theta_propose, const arma::vec deltas_propose);
  
Rcpp::List Gibbs_DINA_HO(const arma::cube& Response, 
                         const arma::cube& Qs, 
                         const arma::mat& test_order, const arma::vec& Test_versions, 
                         const double theta_propose,const arma::vec deltas_propose,
                         const unsigned int chain_length, const unsigned int burn_in);  
  
Rcpp::List parm_update_HO_RT_sep(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                                 arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                 const arma::cube latency, arma::cube& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::vec& tauvar,
                                 const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                                 const arma::mat test_order, const arma::vec Test_versions, const int G_version,
                                 const double theta_propose, const double a_sigma_tau0, const double rate_sigma_tau0, 
                                 const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0);

Rcpp::List Gibbs_DINA_HO_RT_sep(const arma::cube& Response, const arma::cube& Latency,
                                const arma::cube& Qs, 
                                const arma::mat& test_order, const arma::vec& Test_versions, int G_version,
                                const double theta_propose,const arma::vec deltas_propose,
                                const unsigned int chain_length, const unsigned int burn_in);

Rcpp::List parm_update_HO_RT_joint(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                                   arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                   const arma::cube latency, arma::cube& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::mat& Sig,
                                   const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                                   const arma::mat test_order, const arma::vec Test_versions, const int G_version,
                                   const double sig_theta_propose, const arma::mat S, double p,
                                   const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0);  
                                   
Rcpp::List Gibbs_DINA_HO_RT_joint(const arma::cube& Response, const arma::cube& Latency,
                                  const arma::cube& Qs, 
                                  const arma::mat& test_order, const arma::vec& Test_versions, int G_version,
                                  const double sig_theta_propose, const arma::vec deltas_propose,
                                  const unsigned int chain_length, const unsigned int burn_in);


void parm_update_rRUM(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                      arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, 
                      arma::cube& r_stars, arma::mat& pi_stars, const arma::cube Qs, 
                      const arma::cube& responses, arma::cube& X_ijk, arma::cube& Smats, arma::cube& Gmats,
                      const arma::mat& test_order,const arma::vec& Test_versions, const arma::vec& dirich_prior);                                   

Rcpp::List Gibbs_rRUM_indept(const arma::cube& Response, const arma::cube& Qs, const arma::mat& R,
                             const arma::mat& test_order, const arma::vec& Test_versions,
                             const unsigned int chain_length, const unsigned int burn_in);

void parm_update_NIDA_indept(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                             arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, const arma::cube Qs, 
                             const arma::cube& responses, arma::cube& X_ijk, arma::cube& Smats, arma::cube& Gmats,
                             const arma::mat& test_order,const arma::vec& Test_versions, const arma::vec& dirich_prior);

Rcpp::List Gibbs_NIDA_indept(const arma::cube& Response, const arma::cube& Qs, const arma::mat& R,
                             const arma::mat& test_order, const arma::vec& Test_versions,
                             const unsigned int chain_length, const unsigned int burn_in);

void parm_update_DINA_FOHM(unsigned int N,unsigned int J,unsigned int K,unsigned int nClass,
                           unsigned int nT,const arma::cube& Y,const arma::mat& TP,
                           const arma::mat& ETA,arma::vec& ss,arma::vec& gs,arma::mat& CLASS,
                           arma::vec& pi,arma::mat& Omega);

Rcpp::List Gibbs_DINA_FOHM(const arma::cube& Response,const arma::cube& Qs,
                           const arma::mat& test_order, const arma::vec& Test_versions,
                           const unsigned int chain_length, const unsigned int burn_in);

Rcpp::List hmcdm(const arma::cube Y_real_array, const arma::mat Q_matrix, 
                         const std::string model, const arma::mat& test_order, const arma::vec& Test_versions,
                         const unsigned int chain_length, const unsigned int burn_in,
                         const int G_version,
                         const double theta_propose, 
                         const Rcpp::Nullable<arma::cube> Latency_array,
                         const Rcpp::Nullable<Rcpp::NumericVector> deltas_propose,
                         const Rcpp::Nullable<Rcpp::NumericMatrix> R);

#endif