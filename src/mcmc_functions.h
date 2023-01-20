#ifndef MCMC_FUNCTIONS_H
#define MCMC_FUNCTIONS_H

Rcpp::List parm_update_HO_g(const arma::cube Design_array, arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                            const arma::cube response, arma::mat& itempars, const Rcpp::List Q_examinee,
                            const double theta_propose, const arma::vec deltas_propose,
                            const arma::mat Q_matrix);

Rcpp::List Gibbs_DINA_HO_g(const arma::cube& Response, 
                           const arma::mat& Q_matrix,
                           const arma::cube& Design_array,
                           const double theta_propose,const arma::vec deltas_propose,
                           const unsigned int chain_length, const unsigned int burn_in);

Rcpp::List parm_update_HO_RT_sep_g(const arma::cube Design_array, 
                                   arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                   const arma::cube latency, arma::mat& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::vec& tauvar,
                                   const arma::cube response, arma::mat& itempars, const arma::mat Q_matrix, const Rcpp::List Q_examinee,
                                   const int G_version,
                                   const double theta_propose, const double a_sigma_tau0, const double rate_sigma_tau0, 
                                   const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0);

Rcpp::List Gibbs_DINA_HO_RT_sep_g(const arma::cube& Response, const arma::cube& Latency,
                                  const arma::mat& Q_matrix, 
                                  const arma::cube& Design_array, int G_version,
                                  const double theta_propose,const arma::vec deltas_propose,
                                  const unsigned int chain_length, const unsigned int burn_in);

Rcpp::List parm_update_HO_RT_joint_g(const arma::cube& Design_array,
                                     arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                     const arma::cube latency, arma::mat& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::mat& Sig,
                                     const arma::cube response, arma::mat& itempars, const arma::mat Q_matrix, const Rcpp::List Q_examinee,
                                     const int G_version,
                                     const double sig_theta_propose, const arma::mat S, double p,
                                     const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0
);

Rcpp::List Gibbs_DINA_HO_RT_joint_g(const arma::cube& Response, const arma::cube& Latency,
                                    const arma::mat& Q_matrix, 
                                    const arma::cube& Design_array, int G_version,
                                    const double sig_theta_propose, const arma::vec deltas_propose,
                                    const unsigned int chain_length, const unsigned int burn_in);

void parm_update_rRUM_g(const arma::cube& Design_array,
                        arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, 
                        arma::mat& r_stars, arma::vec& pi_stars, const arma::mat Q_matrix, 
                        const arma::cube& responses, arma::cube& X_ijk, arma::mat& Smats, arma::mat& Gmats,
                        const arma::vec& dirich_prior);

Rcpp::List Gibbs_rRUM_indept_g(const arma::cube& Response, const arma::mat& Q_matrix, const arma::mat& R,
                               const arma::cube& Design_array,
                               const unsigned int chain_length, const unsigned int burn_in);

void parm_update_NIDA_indept_g(const arma::cube& Design_array,
                               arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, const arma::mat Q_matrix, 
                               const arma::cube& responses, arma::cube& X_ijk, arma::mat& Smats, arma::mat& Gmats,
                               const arma::vec& dirich_prior);

Rcpp::List Gibbs_NIDA_indept_g(const arma::cube& Response, const arma::mat& Q_matrix, const arma::mat& R,
                               const arma::cube& Design_array,
                               const unsigned int chain_length, const unsigned int burn_in);

void parm_update_DINA_FOHM(unsigned int N,unsigned int J,unsigned int K,unsigned int nClass,
                           unsigned int nT,const arma::cube& Y,const arma::mat& TP,
                           const arma::mat& ETA,arma::vec& ss,arma::vec& gs,arma::mat& CLASS,
                           arma::vec& pi,arma::mat& Omega);

Rcpp::List Gibbs_DINA_FOHM_g(const arma::cube& Response,const arma::mat& Q_matrix,
                             const arma::cube& Design_array,
                             const unsigned int chain_length, const unsigned int burn_in);

Rcpp::List hmcdm(const arma::cube Response, const arma::mat Q_matrix, 
                 const std::string model, 
                 const Rcpp::Nullable<arma::cube> Design_array,
                 const Rcpp::Nullable<arma::mat> Test_order,
                 const Rcpp::Nullable<arma::vec> Test_versions,
                 const unsigned int chain_length, const unsigned int burn_in,
                 const int G_version, 
                 const double theta_propose, 
                 const Rcpp::Nullable<arma::cube> Latency_array,
                 const Rcpp::Nullable<Rcpp::NumericVector> deltas_propose,
                 const Rcpp::Nullable<Rcpp::NumericMatrix> R);



#endif