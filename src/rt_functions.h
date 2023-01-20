#ifndef RT_FUNCTIONS_H
#define RT_FUNCTIONS_H

arma::cube J_incidence_cube(const arma::mat& test_order, const arma::cube& Qs);

arma::cube J_incidence_cube_g(const arma::mat& Q_matrix, const arma::cube& Design_array);

arma::vec G2vec_efficient(const arma::cube& ETA, const arma::cube& J_incidence, const arma::cube& alphas_i, 
                          int test_version_i, const arma::mat test_order, unsigned int t);
                          
arma::vec G2vec_efficient_g(const arma::mat& ETA, const arma::cube& J_incidence, const arma::cube& alphas_i, 
                            unsigned int t, const arma::mat& Q_matrix, const arma::cube& Design_array, unsigned int i);

arma::cube sim_RT(const arma::cube& alphas, const arma::mat& Q_matrix, const arma::cube& Design_array, 
                  const arma::mat& RT_itempars, const arma::vec& taus, double phi, int G_version);

double dLit(const arma::vec& G_it, const arma::vec& L_it, const arma::mat& RT_itempars_it, 
            double tau_i, double phi);                  

#endif