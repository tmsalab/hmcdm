#ifndef TRANS_FUNCTIONS_H
#define TRANS_FUNCTIONS_H

arma::cube simulate_alphas_HO_sep(const arma::vec& lambdas, const arma::vec& thetas, const arma::mat& alpha0s,
                                  const Rcpp::List& Q_examinee, const unsigned int T, const unsigned int Jt);
                                  
double pTran_HO_sep(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                    const arma::mat& Q_i, unsigned int Jt, unsigned int t);

arma::cube simulate_alphas_HO_joint(const arma::vec& lambdas, const arma::vec& thetas, const arma::mat& alpha0s,
                                    const Rcpp::List& Q_examinee, const unsigned int T, const unsigned int Jt);

double pTran_HO_joint(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& lambdas, double theta_i,
                      const arma::mat& Q_i, unsigned int Jt, unsigned int t);

arma::cube simulate_alphas_indept(const arma::vec taus, const arma::mat& alpha0s, const unsigned int T, const arma::mat& R);

double pTran_indept(const arma::vec& alpha_prev, const arma::vec& alpha_post, const arma::vec& taus,const arma::mat& R);

arma::cube simulate_alphas_FOHM(const arma::mat& Omega,const arma::mat& alpha0s,unsigned int T);

arma::mat rAlpha(const arma::mat& Omega,unsigned int N,unsigned int T, const arma::vec& alpha1);

arma::mat rOmega(const arma::mat& TP);  
  
  
#endif