#ifndef RT_FUNCTIONS_H
#define RT_FUNCTIONS_H

arma::cube J_incidence_cube(const arma::mat& test_order, const arma::cube& Qs);

arma::vec G2vec_efficient(const arma::cube& ETA, const arma::cube& J_incidence, const arma::cube& alphas_i, 
                          int test_version_i, const arma::mat test_order, unsigned int t);
                          
arma::cube sim_RT(const arma::cube& alphas, const arma::cube& RT_itempars, const arma::cube& Qs,
                  const arma::vec& taus, double phi, const arma::cube ETA, int G_version,
                  const arma::mat& test_order, arma::vec Test_versions);
                  
double dLit(const arma::vec& G_it, const arma::vec& L_it, const arma::mat& RT_itempars_it, 
            double tau_i, double phi);                  



#endif