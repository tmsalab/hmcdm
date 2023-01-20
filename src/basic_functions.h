#ifndef BASIC_FUNCTIONS_H
#define BASIC_FUNCTIONS_H

arma::vec bijectionvector(unsigned int K);

arma::vec inv_bijectionvector(unsigned int K,double CL);

arma::mat rwishart(unsigned int df, const arma::mat& S);

arma::mat rinvwish(unsigned int df, const arma::mat& Sig);

double rmultinomial(const arma::vec& ps);

arma::vec rDirichlet(const arma::vec& deltas);

double dmvnrm(arma::vec x,arma::vec mean, arma::mat sigma,  bool logd);

arma::vec rmvnrm(arma::vec mu, arma::mat sigma);

arma::mat random_Q(unsigned int J,unsigned int K);

arma::mat ETAmat(unsigned int K,unsigned int J,const arma::mat& Q);

arma::mat TPmat(unsigned int K);

arma::mat crosstab(const arma::vec& V1,const arma::vec& V2,const arma::mat& TP, unsigned int nClass,unsigned int col_dim);
                   
arma::cube resp_miss(const arma::cube& Responses, const arma::mat& test_order, const arma::vec& Test_versions);                   
arma::mat OddsRatio(unsigned int N,unsigned int J,const arma::mat& Yt);

int getMode(arma::vec sorted_vec, int size);

arma::cube Sparse2Dense(const arma::cube Y_real_array, const arma::mat& test_order, const arma::vec& Test_versions);
arma::cube Dense2Sparse(const arma::cube Y_sim,const arma::mat& test_order,const arma::vec& Test_versions);

arma::cube Mat2Array(const arma::mat Q_matrix, unsigned int T);
arma::mat Array2Mat(const arma::cube r_stars);

Rcpp::List Q_list_g(const arma::mat Q_matrix, const arma::cube Design_array);

arma::cube design_array(const arma::mat Test_order, const arma::vec Test_versions, const double Jt);

#endif