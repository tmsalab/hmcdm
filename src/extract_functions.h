#ifndef EXTRACT_FUNCTIONS_H
#define EXTRACT_FUNCTIONS_H

Rcpp::List point_estimates_learning(const Rcpp::List output, const std::string model, const unsigned int N,
                                    const unsigned int Jt, const unsigned int K, const unsigned int T,
                                    bool alpha_EAP);

Rcpp::List Learning_fit(const Rcpp::List output, const std::string model,
                        const Rcpp::List Response_list, const Rcpp::List Q_list,
                        const arma::mat test_order, const arma::vec Test_versions,
                        const Rcpp::Nullable<Rcpp::List> Q_examinee,
                        const Rcpp::Nullable<Rcpp::List> Latency_list, const int G_version,
                        const Rcpp::Nullable<Rcpp::NumericMatrix> R);

#endif