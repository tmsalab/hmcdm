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



#endif