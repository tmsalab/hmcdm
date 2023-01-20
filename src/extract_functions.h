#ifndef EXTRACT_FUNCTIONS_H
#define EXTRACT_FUNCTIONS_H

Rcpp::List point_estimates_learning(const Rcpp::List output, const std::string model, const unsigned int N,
                                    const unsigned int K, const unsigned int T,
                                    bool alpha_EAP);

Rcpp::List Learning_fit_g(const Rcpp::List output, const std::string model,
                          const arma::cube Y_real_array, const arma::mat Q_matrix,
                          const arma::cube Design_array,
                          const Rcpp::Nullable<Rcpp::List> Q_examinee,
                          const Rcpp::Nullable<arma::cube> Latency_array, 
                          const int G_version,
                          const Rcpp::Nullable<Rcpp::NumericMatrix> R);
  
#endif