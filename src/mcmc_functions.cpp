#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "resp_functions.h"
#include "rt_functions.h"
#include "trans_functions.h"
#include "mcmc_functions.h"

// ----------------------------- MCMC Functions --------------------------------------------------------------
// Single iteration parameter updates, full Gibbs sampler, DIC computation 
// -----------------------------------------------------------------------------------------------------------



// [[Rcpp::export]]
Rcpp::List parm_update_HO(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                          arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                          const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                          const arma::mat test_order, const arma::vec Test_versions, 
                          const double theta_propose, const arma::vec deltas_propose){
  arma::cube ETA(Jt, (pow(2,K)), T);
  arma::vec CLASS_0(N);
  for(unsigned int t = 0; t<T; t++){
    ETA.slice(t) = ETAmat(K,Jt, Qs.slice(t));
  }
  
  double post_new, post_old;
  double theta_i_new;
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  
  for(unsigned int i = 0; i<N; i++){
    int test_version_i = Test_versions(i)-1;
    double theta_i = thetas(i);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      int test_block_it = test_order(test_version_i,t)-1;
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        likelihood_Y(cc) = pYit_DINA(ETA.slice(test_block_it).col(cc), response.slice(t).row(i).t(),itempars.slice(test_block_it));
      }
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      // initial time point
      if(t == 0){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % pi % ptranspost;
        probs = probs/arma::sum(probs);
        double tmp = rmultinomial(probs);
        CLASS_0(i) = tmp;
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // middle points
      if(t > 0 && t < (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probabilities
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % ptransprev % ptranspost;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // last time point
      if(t == (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probs
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % ptransprev;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
    }
    // update theta_i
    theta_i_new = R::rnorm(theta_i,theta_propose);
    
    // The prior of for theta, N(0,1)
    post_old = std::log(R::dnorm(theta_i, 0, 1, false));
    post_new = std::log(R::dnorm(theta_i_new, 0, 1, false));
    
    // multiply prior by trans prob at time t>0
    for(unsigned int t = 1; t<T; t++){
      post_old += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Jt,(t-1)));
      post_new += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i_new,Q_i,Jt,(t-1)));
    }
    ratio = exp(post_new - post_old);
    u = R::runif(0,1);
    if(u < ratio){
      thetas(i) = theta_i_new;
      accept_theta(i) = 1;
    }
  }
  
  // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  // update lambdas
  arma::vec accept_lambdas = arma::zeros<arma::vec>(lambdas.n_elem);
  for(unsigned int h = 0; h<lambdas.n_elem; h++){
    // reset tmp
    arma::vec tmp = lambdas;
    tmp(h) = R::runif((lambdas(h)-deltas_propose(h)), (lambdas(h)+deltas_propose(h)));
    if(h == 0){
      post_old = R::dnorm(lambdas(h),0,0.5,true);
      post_new = R::dnorm(tmp(h),0,0.5,true);
    }else{
      if (h == 1) {
        post_old = R::dlnorm(lambdas(h), 0, .5, 1);
        post_new = R::dlnorm(tmp(h), 0, .5, 1);
      }
      else {
        post_old = R::dlnorm(lambdas(h), -0.5, .6, 1);
        post_new = R::dlnorm(tmp(h), -0.5, .6, 1);
      }
    }
    
    for(unsigned int i = 0; i < N; i++){
      for(unsigned int t = 0; t < (T-1); t++){
        post_old += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas, thetas(i), Q_examinee[i], Jt, t));
        post_new += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          tmp, thetas(i), Q_examinee[i], Jt, t));
      }
    }
    
    ratio = exp(post_new-post_old);
    u = R::runif(0,1);
    if(u < ratio){
      lambdas = tmp;
      accept_lambdas(h) = 1;
    }
  }
  
  // update s, g, alpha, gamma for items, and save the aggregated coefficients for the posterior of phi
  double as, bs, ag, bg, pg, ps, ug, us;
  for(unsigned int block = 0; block < T; block++){
    arma::mat Res_block(N, Jt);
    arma::mat RT_block(N, Jt);
    arma::mat Q_current = Qs.slice(block);
    arma::vec Class(N);
    arma::vec eta_j(N);
    arma::mat ETA_block(N,Jt);
    for(unsigned int i = 0; i < N; i++){
      // find the time point at which i received this block
      int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(block+1)));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.row(i) = response.slice(t_star).row(i);
      arma::vec alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      
      for(unsigned int j = 0; j < Jt; j++){
        ETA_block(i,j) = ETA(j,Class(i),block);
      }
    }
    
    for(unsigned int j = 0; j < Jt; j++){
      eta_j = ETA_block.col(j);
      
      // sample the response model parameters
      us = R::runif(0, 1);
      ug = R::runif(0, 1);
      // get posterior a, b for sj and gj
      as = arma::sum(eta_j % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      bs = arma::sum(eta_j % Res_block.col(j)) + 1;
      ag = arma::sum((arma::ones<arma::vec>(N)-eta_j) % Res_block.col(j)) + 1;
      bg = arma::sum((arma::ones<arma::vec>(N)-eta_j) % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      // update g based on s on previous iteration
      pg = R::pbeta(1.0 - itempars(j,0,block), ag, bg, 1, 0);
      itempars(j,1,block) = R::qbeta(ug*pg, ag, bg, 1, 0);
      // update s based on current g
      ps = R::pbeta(1.0 - itempars(j,1,block), as, bs, 1, 0);
      itempars(j,0,block) = R::qbeta(us*ps, as, bs, 1, 0);
    }
  }
  
  
  // return the results
  return Rcpp::List::create(Rcpp::Named("accept_theta",accept_theta),
                            Rcpp::Named("accept_lambdas",accept_lambdas)
  );
}


// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_HO(const arma::cube& Response, 
                         const arma::cube& Qs, const Rcpp::List Q_examinee,
                         const arma::mat& test_order, const arma::vec& Test_versions, 
                         const double theta_propose,const arma::vec deltas_propose,
                         const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  
  //std::srand(std::time(0));//use current time as seed for random generator
  
  // initialize parameters
  arma::vec lambdas_init(4),tauvar_init(1);
  lambdas_init(0) = R::rnorm(0,1);
  lambdas_init(1) = R::runif(0,1);
  lambdas_init(2) = R::runif(0,1);
  lambdas_init(3) = R::runif(0,1);
  
  arma::vec thetas_init(N);
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  tauvar_init(0) = R::runif(1, 1.5); // initial value for the variance of taus
  
  for(unsigned int i = 0; i < N; i++){
    thetas_init(i,0) = R::rnorm(0, 1);
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  arma::cube Alphas_init = simulate_alphas_HO_sep(lambdas_init,thetas_init,Alphas_0_init,
                                                  Q_examinee, T, Jt);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  
  arma::cube itempars_init = .3 * arma::randu<arma::cube>(Jt,2,T);
  itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) =
    itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) % (1.-itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)));
  
  // Create objects for storage
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat ss(J,(chain_length-burn_in));
  arma::mat gs(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat thetas(N,(chain_length-burn_in));
  arma::mat lambdas(4,(chain_length-burn_in));
  double accept_rate_theta = 0;
  arma::vec accept_rate_lambdas = arma::zeros<arma::vec>(4);
  
  double tmburn;//,deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for (unsigned int tt = 0; tt < chain_length; tt++) {
    Rcpp::List tmp = parm_update_HO(N,Jt,K,T,Alphas_init,pi_init,lambdas_init,thetas_init,Response,
                                    itempars_init,Qs,Q_examinee,test_order,Test_versions,
                                    theta_propose,deltas_propose);
    if (tt >= burn_in) {
      tmburn = tt - burn_in;
      for (unsigned int t = 0; t < T; t++) {
        Trajectories_mat.cols(K*t, (K*(t + 1) - 1)) = Alphas_init.slice(t);
        ss.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(0);
        gs.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(1);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      thetas.col(tmburn) = thetas_init;
      lambdas.col(tmburn) = lambdas_init;
      accept_theta_vec = Rcpp::as<arma::vec>(tmp[0]);
      accept_lambdas_vec = Rcpp::as<arma::vec>(tmp[1]);
      m_accept_theta = arma::mean(accept_theta_vec);
      accept_rate_theta = (accept_rate_theta*tmburn + m_accept_theta) / (tmburn + 1.);
      accept_rate_lambdas = (accept_rate_lambdas*tmburn + accept_lambdas_vec) / (tmburn + 1.);
      
      
      
    }
    
    if (tt % 1000 == 0) {
      Rcpp::Rcout << tt << std::endl;
    }
  }
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",ss),
                            Rcpp::Named("gs",gs),
                            Rcpp::Named("pis", pis),
                            Rcpp::Named("thetas",thetas),
                            Rcpp::Named("lambdas",lambdas),
                            Rcpp::Named("accept_rate_theta",accept_rate_theta),
                            Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas)
                              // Rcpp::Named("accept_rate_tau", accept_rate_tau),
                              // Rcpp::Named("time_pp", time_pp),
                              // Rcpp::Named("res_pp", res_pp),
                              // Rcpp::Named("Deviance",Deviance),
                              // Rcpp::Named("D_DINA", Deviance_DINA),
                              // Rcpp::Named("D_tran",Deviance_tran)
  );
}












// [[Rcpp::export]]
Rcpp::List parm_update_HO_RT_sep(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                                 arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                 const arma::cube latency, arma::cube& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::vec& tauvar,
                                 const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                                 const arma::mat test_order, const arma::vec Test_versions, const int G_version,
                                 const double theta_propose, const double a_sigma_tau0, const double rate_sigma_tau0, 
                                 const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0
){
  double phi = phi_vec(0);
  double tau_sig = tauvar(0);
  arma::cube ETA(Jt, (pow(2,K)), T);
  arma::vec CLASS_0(N);
  for(unsigned int t = 0; t<T; t++){
    ETA.slice(t) = ETAmat(K,Jt, Qs.slice(t));
  }
  arma::cube J_incidence = J_incidence_cube(test_order,Qs);
  
  double post_new, post_old;
  arma::vec thetatau_i_old(2);
  arma::vec thetatau_i_new(2);
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  arma::vec accept_tau = arma::zeros<arma::vec>(N);
  
  for(unsigned int i = 0; i<N; i++){
    int test_version_i = Test_versions(i)-1;
    double theta_i = thetas(i);
    double tau_i = taus(i);
    arma::vec G_it = arma::zeros<arma::vec>(Jt);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      int test_block_it = test_order(test_version_i,t)-1;
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        likelihood_Y(cc) = pYit_DINA(ETA.slice(test_block_it).col(cc), response.slice(t).row(i).t(),itempars.slice(test_block_it));
      }
      // likelihood of RT (time dependent)
      arma::vec likelihood_L = arma::ones<arma::vec>(pow(2,K));
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      int test_block_itt;
      // initial time point
      if(t == 0){
        
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          
          // likelihood of RT
          if(G_version !=3){
            if(G_version==1 ){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =0; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
            
            
          }else{
            likelihood_L(cc)=1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % pi % ptranspost;
        probs = probs/arma::sum(probs);
        double tmp = rmultinomial(probs);
        CLASS_0(i) = tmp;
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // middle points
      if(t > 0 && t < (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probabilities
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          
          // likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =t; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
          }else{
            likelihood_L(cc)=1;
          }
          
          
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev % ptranspost;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      
      // last time point
      if(t == (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probs
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          // Likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
            }
            if(G_version==2){
              G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                     test_order,t);
            }
            
            
            
            likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                         tau_i,phi);
          }else{
            
            likelihood_L(cc) =1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
    }
    // update theta_i
    
    thetatau_i_new(0)= R::rnorm(theta_i,theta_propose);
    
    // The prior of for theta, N(0,1)
    
    post_old = std::log(R::dnorm(theta_i, 0, 1, false));
    post_new = std::log(R::dnorm(thetatau_i_new(0), 0, 1, false));
    
    // multiply prior by trans prob at time t>0
    for(unsigned int t = 1; t<T; t++){
      
      post_old += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Jt,(t-1)));
      post_new += std::log(pTran_HO_sep(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,thetatau_i_new(0),Q_i,Jt,(t-1)));
    }
    ratio = exp(post_new - post_old);
    u = R::runif(0,1);
    if(u < ratio){
      thetas(i) = thetatau_i_new(0);
      accept_theta(i) = 1;
    }
    
    // update tau_i, Gbbs, draw the tau_i from the posterial distribution, which is still normal
    
    double num = 0;
    double denom = 0;
    test_version_i = Test_versions(i) - 1;
    for (unsigned int t = 0; t<T; t++) {
      int	test_block_it = test_order(test_version_i, t) - 1;
      
      if (G_version == 1) {
        double class_it = arma::dot(alphas.slice(t).row(i).t(), vv);
        G_it = ETA.slice(test_block_it).col(class_it);
      }
      if (G_version == 2) {
        G_it = G2vec_efficient(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                               test_order, t);
      }
      if(G_version==3){
        
        G_it = arma::ones<arma::vec>(Jt);
        arma::vec y(Jt);
        y.fill((t+1.)/T);
        G_it =G_it % y;
      }
      
      
      for (unsigned int j = 0; j<Jt; j++) {
        //add practice*a of (i,j,block) to num and denom of phi
        num += (log(latency(i, j, t))-RT_itempars(j, 1, test_block_it)+ phi*G_it(j))*pow(RT_itempars(j, 0, test_block_it), 2);
        denom += pow(RT_itempars(j, 0, test_block_it), 2);
      }
    }
    // sample tau_i
    double mu_tau, sigma_tau;
    mu_tau = -num / (denom + (1/tau_sig));
    sigma_tau = sqrt(1. / (denom + (1/tau_sig)));
    taus(i) = R::rnorm(mu_tau, sigma_tau);
    
  }
  
  //update the variance for tau from inverse-gamma distribution
  // check this inverse
  
  double a_sigma_tau = a_sigma_tau0 + N / 2.;
  double b_sigma_tau = 1. / (rate_sigma_tau0 + arma::dot(taus.t(), taus) / 2.);
  tauvar(0) = 1. / R::rgamma(a_sigma_tau, b_sigma_tau);
  
  // // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  // update lambdas
  arma::vec accept_lambdas = arma::zeros<arma::vec>(lambdas.n_elem);
  for(unsigned int h = 0; h<lambdas.n_elem; h++){
    // reset tmp
    arma::vec tmp = lambdas;
    tmp(h) = R::runif((lambdas(h)-deltas_propose(h)), (lambdas(h)+deltas_propose(h)));
    if(h == 0){
      post_old = R::dnorm(lambdas(h),0,0.5,true);
      post_new = R::dnorm(tmp(h),0,0.5,true);
    }else{
      if (h == 1) {
        post_old = R::dlnorm(lambdas(h), 0, .5, 1);
        post_new = R::dlnorm(tmp(h), 0, .5, 1);
      }
      else {
        post_old = R::dlnorm(lambdas(h), -0.5, .6, 1);
        post_new = R::dlnorm(tmp(h), -0.5, .6, 1);
      }
    }
    
    for(unsigned int i = 0; i < N; i++){
      for(unsigned int t = 0; t < (T-1); t++){
        post_old += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas, thetas(i), Q_examinee[i], Jt, t));
        post_new += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          tmp, thetas(i), Q_examinee[i], Jt, t));
      }
    }
    
    ratio = exp(post_new-post_old);
    u = R::runif(0,1);
    if(u < ratio){
      lambdas = tmp;
      accept_lambdas(h) = 1;
    }
  }
  
  // update s, g, alpha, gamma for items, and save the aggregated coefficients for the posterior of phi
  double as, bs, ag, bg, pg, ps, ug, us;
  double a_alpha, scl_alpha, mu_gamma, sd_gamma, alpha_sqr;
  double tau_i;
  arma::cube Gs(N, Jt,T);
  for(unsigned int block = 0; block < T; block++){
    arma::mat Res_block(N, Jt);
    arma::mat RT_block(N, Jt);
    arma::mat Q_current = Qs.slice(block);
    arma::vec Class(N);
    arma::vec eta_j(N);
    arma::mat ETA_block(N,Jt);
    for(unsigned int i = 0; i < N; i++){
      // find the time point at which i received this block
      int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(block+1)));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.row(i) = response.slice(t_star).row(i);
      RT_block.row(i) = latency.slice(t_star).row(i);
      arma::vec alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      
      for(unsigned int j = 0; j < Jt; j++){
        ETA_block(i,j) = ETA(j,Class(i),block);
        if(G_version == 1){
          Gs(i,j,block) = ETA_block(i,j);
        }
        if(G_version==3){
          Gs(i,j,block)= (t_star+1.)/T;
        }
      }
      if(G_version == 2){
        Gs.slice(block).row(i) = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),
                 (Test_versions(i)-1),test_order,t_star).t();
      }
      
      
    }
    
    for(unsigned int j = 0; j < Jt; j++){
      eta_j = ETA_block.col(j);
      
      // sample the response model parameters
      us = R::runif(0, 1);
      ug = R::runif(0, 1);
      // get posterior a, b for sj and gj
      as = arma::sum(eta_j % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      bs = arma::sum(eta_j % Res_block.col(j)) + 1;
      ag = arma::sum((arma::ones<arma::vec>(N)-eta_j) % Res_block.col(j)) + 1;
      bg = arma::sum((arma::ones<arma::vec>(N)-eta_j) % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      // update g based on s on previous iteration
      pg = R::pbeta(1.0 - itempars(j,0,block), ag, bg, 1, 0);
      itempars(j,1,block) = R::qbeta(ug*pg, ag, bg, 1, 0);
      // update s based on current g
      ps = R::pbeta(1.0 - itempars(j,1,block), as, bs, 1, 0);
      itempars(j,0,block) = R::qbeta(us*ps, as, bs, 1, 0);
      
      // sample the RT model parameters
      // scl_tmp: (log(L_ij) + tau_i + phi * G_ij -gamma_j)^2
      // mu_tmp: log(L_it) + tau_i + phi * G_ij
      arma::vec scl_tmp(N);
      arma::vec mu_tmp(N);
      for(unsigned int i = 0; i < N; i++){
        tau_i = taus(i);
        scl_tmp(i) = pow((log(RT_block(i,j))+tau_i+phi*Gs(i,j,block)-RT_itempars(j,1,block)),2);
        mu_tmp(i) = log(RT_block(i,j))+tau_i+phi*Gs(i,j,block);
      }
      // update alpha_j based on previous gamma_j
      a_alpha = a_alpha0 + N/2;
      // note: the derivation we have corresponds to the rate of gamma, need to take recip for scl
      scl_alpha = 1./(rate_alpha0 + 1./2 * arma::sum(scl_tmp));
      alpha_sqr = R::rgamma(a_alpha,scl_alpha);
      RT_itempars(j,0,block) = sqrt(alpha_sqr);
      // update gamma_j based on current alpha_j
      mu_gamma = (alpha_sqr*arma::sum(mu_tmp))/(N*alpha_sqr + 1);
      sd_gamma = sqrt(1./(N*alpha_sqr + 1));
      RT_itempars(j,1,block) = R::rnorm(mu_gamma, sd_gamma);
    }
  }
  
  
  // update phi
  double num = 0;
  double denom = 0;
  unsigned int test_version_i, block_it;
  for(unsigned int i = 0; i<N; i++){
    tau_i = taus(i);
    test_version_i = Test_versions(i)-1;
    for(unsigned int t = 0; t<T; t++){
      block_it = test_order(test_version_i,t)-1;
      for(unsigned int j = 0; j<Jt; j++){
        //add practice*a of (i,j,block) to num and denom of phi
        num += pow(RT_itempars(j,0,block_it),2) * Gs(i,j,block_it)*(log(latency(i,j,t))-RT_itempars(j,1,block_it)+tau_i);
        denom += pow(RT_itempars(j,0,block_it),2)* pow(Gs(i,j,block_it),2);
      }
    }
  }
  
  double mu_phi, sigma_phi;
  mu_phi = -num/(denom+1.);
  sigma_phi = sqrt(1./(denom+1.));
  phi_vec(0) = R::rnorm(mu_phi,sigma_phi);
  
  
  
  
  
  // return the results
  return Rcpp::List::create(Rcpp::Named("accept_theta",accept_theta),
                            Rcpp::Named("accept_lambdas",accept_lambdas)
  );
}


// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_HO_RT_sep(const arma::cube& Response, const arma::cube& Latency,
                                const arma::cube& Qs, const Rcpp::List Q_examinee,
                                const arma::mat& test_order, const arma::vec& Test_versions, int G_version,
                                const double theta_propose,const arma::vec deltas_propose,
                                const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  
  //std::srand(std::time(0));//use current time as seed for random generator
  
  // initialize parameters
  arma::vec lambdas_init(4),tauvar_init(1);
  lambdas_init(0) = R::rnorm(0,1);
  lambdas_init(1) = R::runif(0,1);
  lambdas_init(2) = R::runif(0,1);
  lambdas_init(3) = R::runif(0, 1);
  
  //arma::mat Sig_init = arma::eye<arma::mat>(2,2);
  arma::mat thetatau_init(N,2);
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  tauvar_init(0) = R::runif(1, 1.5); // initial value for the variance of taus
  
  for(unsigned int i = 0; i < N; i++){
    thetatau_init(i,0) = R::rnorm(0, 1);
    thetatau_init(i,1)= R::rnorm(0,tauvar_init(0));
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  arma::vec thetas_init = thetatau_init.col(0);
  arma::vec taus_init = thetatau_init.col(1);
  arma::cube Alphas_init = simulate_alphas_HO_sep(lambdas_init,thetatau_init.col(0),Alphas_0_init,
                                                  Q_examinee, T, Jt);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec phi_init(1);
  phi_init(0) = R::runif(0,1);
  //phi_init(0) =0;
  
  arma::cube itempars_init = .3 * arma::randu<arma::cube>(Jt,2,T);
  itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) =
    itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) % (1.-itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)));
  arma::cube RT_itempars_init(Jt,2,T);
  RT_itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)) = 2.+2.*arma::randu<arma::cube>(Jt,1,T);
  RT_itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) = arma::randn<arma::cube>(Jt,1,T); // why take exponetial part?
  
  //double p = 3.;
  //
  // Create objects for storage
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat ss(J,(chain_length-burn_in));
  arma::mat gs(J,(chain_length-burn_in));
  arma::mat RT_as(J,(chain_length-burn_in));
  arma::mat RT_gammas(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat thetas(N,(chain_length-burn_in));
  arma::mat taus(N,(chain_length-burn_in));
  arma::mat lambdas(4,(chain_length-burn_in));
  arma::vec phis(chain_length-burn_in);
  //arma::cube Sigs(2,2,(chain_length-burn_in));
  arma::vec tauvar(chain_length - burn_in);
  double accept_rate_theta = 0;
  //double accept_rate_tau =0;
  arma::vec accept_rate_lambdas = arma::zeros<arma::vec>(4);
  // arma::cube time_pp(N,T,(chain_length-burn_in));
  // arma::cube res_pp(N,T,(chain_length-burn_in));
  // arma::vec Deviance(chain_length-burn_in);
  // arma::vec Deviance_DINA(chain_length-burn_in);
  // arma::vec Deviance_RT(chain_length-burn_in);
  // arma::vec Deviance_tran(chain_length-burn_in);
  
  double tmburn;//,deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec,accept_tau_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for (unsigned int tt = 0; tt < chain_length; tt++) {
    Rcpp::List tmp = parm_update_HO_RT_sep(N, Jt, K, T, Alphas_init, pi_init, lambdas_init,
                                           thetas_init, Latency, RT_itempars_init, taus_init,
                                           phi_init, tauvar_init, Response, itempars_init, Qs,
                                           Q_examinee, test_order, Test_versions, G_version,
                                           theta_propose, 2.5, 1., deltas_propose, 1., 1.);
    if (tt >= burn_in) {
      tmburn = tt - burn_in;
      for (unsigned int t = 0; t < T; t++) {
        Trajectories_mat.cols(K*t, (K*(t + 1) - 1)) = Alphas_init.slice(t);
        ss.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(0);
        gs.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(1);
        RT_as.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(0);
        RT_gammas.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(1);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      thetas.col(tmburn) = thetas_init;
      taus.col(tmburn) = taus_init;
      lambdas.col(tmburn) = lambdas_init;
      phis(tmburn) = phi_init(0);
      tauvar(tmburn) = tauvar_init(0);
      
      accept_theta_vec = Rcpp::as<arma::vec>(tmp[0]);
      accept_lambdas_vec = Rcpp::as<arma::vec>(tmp[1]);
      m_accept_theta = arma::mean(accept_theta_vec);
      accept_rate_theta = (accept_rate_theta*tmburn + m_accept_theta) / (tmburn + 1.);
      accept_rate_lambdas = (accept_rate_lambdas*tmburn + accept_lambdas_vec) / (tmburn + 1.);
      // ETA = Rcpp::as<arma::cube>(tmp[2]);
      // J_incidence = Rcpp::as<arma::cube>(tmp[3]);
      // 
      // //Deviance at this iteration (We use the joint distribution)
      // 
      // double tran=0, response=0, time=0;
      // arma::vec G_it = arma::zeros<arma::vec>(Jt);
      // arma::vec vv = bijectionvector(K);
      // 
      // for (unsigned int i = 0; i < N; i++) {
      //   int test_version_i = Test_versions(i) - 1;
      //   for (unsigned int t = 0; t < T; t++) {
      //     
      //     // The transition model part
      //     if (t < (T - 1)) {
      //       tran += std::log(pTran_HO_sep(Alphas_init.slice(t).row(i).t(),
      //                              Alphas_init.slice(t + 1).row(i).t(),
      //                              lambdas_init, thetas_init(i), Q_examinee[i], Jt, t));
      //     }
      //     // The log likelihood from response time model
      //     int test_block_it = test_order(test_version_i, t) - 1;
      //     double class_it = arma::dot(Alphas_init.slice(t).row(i).t(), vv);
      //     if (G_version == 1) {
      //       G_it = ETA.slice(test_block_it).col(class_it);
      //     }
      //     if (G_version == 2) {
      //       G_it = G2vec_efficient(ETA, J_incidence, Alphas_init.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
      //                              test_order, t);
      //     }
      //     if(G_version==3){
      //       
      //       G_it = arma::ones<arma::vec>(Jt);
      //       arma::vec y(Jt);y.fill((t+1.)/T);
      //       G_it =G_it % y;
      //     }
      //     
      //     time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), RT_itempars_init.slice(test_block_it),
      //                           taus_init(i), phis(tmburn)));
      //     
      //     // The loglikelihood from the DINA
      //     
      //     response += std::log(pYit_DINA(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), itempars_init.slice(test_block_it)));
      //     if (t == 0) {
      //       response += std::log(pi_init(class_it));
      //     }
      //     
      //   }
      //   
      //   
      // }
      // 
      // // Calculate the deviance in this iteration 
      // 
      // deviance = response + time + tran;
      // Deviance(tmburn)=-2*deviance;
      // Deviance_DINA(tmburn)=-2*response;
      // Deviance_RT(tmburn)=-2*(time);
      // Deviance_tran(tmburn)=-2*tran;
      // 
      // // ppp
      // // Generate response time
      // arma::cube L_model = sim_RT(Alphas_init, RT_itempars_init, Qs, taus_init, phi_init(0), ETA, G_version,
      //                             test_order, Test_versions);
      // 
      // //Generate the response from DINA model
      // // store the total response time for each block, the total score of each person on each block;
      // arma::cube Res_model = simDINA(Alphas_init, itempars_init, ETA, test_order, Test_versions);
      // 
      // arma::mat L_it_model(N, T), DINA_it_model(N, T);
      // for (unsigned int t = 0; t < T; t++) {
      //   L_it_model.col(t) = arma::sum(L_model.slice(t), 1);
      //   DINA_it_model.col(t) = arma::sum(Res_model.slice(t), 1);
      // }
      // 
      // time_pp.slice(tmburn) = L_it_model;
      // res_pp.slice(tmburn) = DINA_it_model;
      // 
      
      
      
      
    }
    if (tt % 1000 == 0) {
      Rcpp::Rcout << tt << std::endl;
    }
    
  }
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",ss),
                            Rcpp::Named("gs",gs),
                            Rcpp::Named("as",RT_as),
                            Rcpp::Named("gammas",RT_gammas),
                            Rcpp::Named("pis", pis),
                            Rcpp::Named("thetas",thetas),
                            Rcpp::Named("taus",taus),
                            Rcpp::Named("lambdas",lambdas),
                            Rcpp::Named("phis",phis),
                            Rcpp::Named("tauvar", tauvar),
                            Rcpp::Named("accept_rate_theta",accept_rate_theta),
                            Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas)
                              // Rcpp::Named("accept_rate_tau", accept_rate_tau),
                              // Rcpp::Named("time_pp", time_pp),
                              // Rcpp::Named("res_pp", res_pp),
                              // Rcpp::Named("Deviance",Deviance),
                              // Rcpp::Named("D_DINA", Deviance_DINA),
                              // Rcpp::Named("D_RT", Deviance_RT),
                              // Rcpp::Named("D_tran",Deviance_tran)
  );
}





// [[Rcpp::export]]
Rcpp::List parm_update_HO_RT_joint(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                                   arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                   const arma::cube latency, arma::cube& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::mat& Sig,
                                   const arma::cube response, arma::cube& itempars, const arma::cube Qs, const Rcpp::List Q_examinee,
                                   const arma::mat test_order, const arma::vec Test_versions, const int G_version,
                                   const double sig_theta_propose, const arma::mat S, double p,
                                   const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0
){
  double phi = phi_vec(0);
  arma::vec CLASS_0(N);
  arma::cube ETA(Jt, (pow(2,K)), T);
  for(unsigned int t = 0; t<T; t++){
    ETA.slice(t) = ETAmat(K,Jt, Qs.slice(t));
  }
  arma::cube J_incidence = J_incidence_cube(test_order,Qs);
  
  double post_new, post_old;
  double theta_new;
  arma::vec thetatau_i_old(2);
  arma::vec thetatau_i_new(2);
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  arma::vec accept_tau = arma::zeros<arma::vec>(N);  
  for(unsigned int i = 0; i<N; i++){
    int test_version_i = Test_versions(i)-1;
    double theta_i = thetas(i);
    double tau_i = taus(i);
    arma::vec G_it = arma::zeros<arma::vec>(Jt);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      int test_block_it = test_order(test_version_i,t)-1;
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        likelihood_Y(cc) = pYit_DINA(ETA.slice(test_block_it).col(cc), response.slice(t).row(i).t(),itempars.slice(test_block_it));
      }
      // likelihood of RT (time dependent)
      arma::vec likelihood_L = arma::ones<arma::vec>(pow(2,K));
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      int test_block_itt;
      // initial time point
      if(t == 0){
        
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_joint(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          
          // likelihood of RT
          if(G_version !=3){
            if(G_version==1 ){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =0; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
            
            
          }else{
            likelihood_L(cc)=1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % pi % ptranspost;
        probs = probs/arma::sum(probs);
        double tmp = rmultinomial(probs);
        CLASS_0(i) = tmp;
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      // middle points
      if(t > 0 && t < (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probabilities
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_joint(alpha_c,alpha_post,lambdas,theta_i,Q_i,Jt,t);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_joint(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          
          // likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
              likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =t; tt<T; tt++){
                test_block_itt = test_order(test_version_i,tt)-1;
                G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                       test_order,tt);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency.slice(tt).row(i).t(),
                             RT_itempars.slice(test_block_itt),tau_i,phi);
              }
            }
          }else{
            likelihood_L(cc)=1;
          }
          
          
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev % ptranspost;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
      
      // last time point
      if(t == (T-1)){
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // Transition probs
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_joint(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Jt,(t-1));
          // Likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              G_it = ETA.slice(test_block_it).col(cc);
            }
            if(G_version==2){
              G_it = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),test_version_i,
                                     test_order,t);
            }
            
            
            
            likelihood_L(cc) = dLit(G_it,latency.slice(t).row(i).t(),RT_itempars.slice(test_block_it),
                         tau_i,phi);
          }else{
            
            likelihood_L(cc) =1;
          }
        }
        // get the full conditional prob
        arma::vec probs = likelihood_Y % likelihood_L % ptransprev;
        double tmp = rmultinomial(probs/arma::sum(probs));
        alphas.slice(t).row(i) = inv_bijectionvector(K,tmp).t();
      }
    }
    // update theta_i, tau_i
    thetatau_i_old(0) = theta_i;
    thetatau_i_old(1) = tau_i;
    
    theta_new = R::rnorm(theta_i,sig_theta_propose);
    thetatau_i_new = thetatau_i_old;
    thetatau_i_new(0) = theta_new;
    post_old = std::log(dmvnrm(thetatau_i_old,arma::zeros<arma::vec>(2),Sig,false));
    post_new = std::log(dmvnrm(thetatau_i_new,arma::zeros<arma::vec>(2),Sig,false));
    for(unsigned int t = 1; t<T; t++){
      post_old += std::log(pTran_HO_joint(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Jt,(t-1)));
      post_new += std::log(pTran_HO_joint(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_new,Q_i,Jt,(t-1)));
    }
    ratio = exp(post_new - post_old);
    u = R::runif(0,1);
    if(u < ratio){
      thetas(i) = thetatau_i_new(0);
      thetatau_i_old = thetatau_i_new;
      accept_theta(i) = 1;
    }
    
    // update tau_i, Gbbs, draw the tau_i from the posterial distribution, which is still normal
    
    double num = 0;
    double denom = 0;
    test_version_i = Test_versions(i) - 1;
    for (unsigned int t = 0; t<T; t++) {
      int	test_block_it = test_order(test_version_i, t) - 1;
      
      if (G_version == 1) {
        double class_it = arma::dot(alphas.slice(t).row(i).t(), vv);
        G_it = ETA.slice(test_block_it).col(class_it);
      }
      if (G_version == 2) {
        G_it = G2vec_efficient(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                               test_order, t);
      }
      if(G_version==3){
        
        G_it = arma::ones<arma::vec>(Jt);
        arma::vec y(Jt);y.fill((t+1.)/T);
        G_it =G_it % y;
      }
      for (unsigned int j = 0; j<Jt; j++) {
        //add practice*a of (i,j,block) to num and denom of phi
        num += (log(latency(i, j, t))-RT_itempars(j, 1, test_block_it)+ phi*G_it(j))*pow(RT_itempars(j, 0, test_block_it), 2);
        denom += pow(RT_itempars(j, 0, test_block_it), 2);
      }
    }
    // sample tau_i
    double mu_tau, sigma_tau;
    mu_tau = -num / (denom + (1/Sig(1,1)));
    sigma_tau = sqrt(1. / (denom + (1/Sig(1,1))));
    taus(i) = R::rnorm(mu_tau, sigma_tau);
  }
  
  // update Sigma for thetatau
  arma::mat thetatau_mat(N,2);
  thetatau_mat.col(0) = thetas;
  thetatau_mat.col(1) = taus;
  arma::mat S_star = thetatau_mat.t() * thetatau_mat + S;
  unsigned int p_star = p + N;
  Sig = rinvwish(p_star,S_star);
  
  // // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  // update lambdas
  arma::vec accept_lambdas = arma::zeros<arma::vec>(3);
  for(unsigned int h = 0; h<lambdas.n_elem; h++){
    // reset tmp
    arma::vec tmp = lambdas;
    tmp(h) = R::runif((lambdas(h)-deltas_propose(h)), (lambdas(h)+deltas_propose(h)));
    if(h == 0){
      post_old = R::dnorm4(lambdas(h),0,0.5,1);
      post_new = R::dnorm4(tmp(h),0,0.5,1);
    }
    else {
      
      post_old = R::dlnorm(lambdas(h), -0.5, .6, 1);
      post_new = R::dlnorm(tmp(h), -0.5, .6, 1);
      
      
    }
    
    for(unsigned int i = 0; i < N; i++){
      for(unsigned int t = 0; t < (T-1); t++){
        post_old += std::log(pTran_HO_joint(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            lambdas, thetas(i), Q_examinee[i], Jt, t));
        post_new += std::log(pTran_HO_joint(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            tmp, thetas(i), Q_examinee[i], Jt, t));
      }
    }
    
    ratio = exp(post_new-post_old);
    u = R::runif(0,1);
    if(u < ratio){
      lambdas = tmp;
      accept_lambdas(h) = 1;
    }
  }
  
  // update s, g, alpha, gamma for items, and save the aggregated coefficients for the posterior of phi
  double as, bs, ag, bg, pg, ps, ug, us;
  double a_alpha, scl_alpha, mu_gamma, sd_gamma, alpha_sqr;
  double tau_i;
  arma::cube Gs(N, Jt,T);
  for(unsigned int block = 0; block < T; block++){
    arma::mat Res_block(N, Jt);
    arma::mat RT_block(N, Jt);
    arma::mat Q_current = Qs.slice(block);
    arma::vec Class(N);
    arma::vec eta_j(N);
    arma::mat ETA_block(N,Jt);
    for(unsigned int i = 0; i < N; i++){
      // find the time point at which i received this block
      int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(block+1)));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.row(i) = response.slice(t_star).row(i);
      RT_block.row(i) = latency.slice(t_star).row(i);
      arma::vec alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      
      for(unsigned int j = 0; j < Jt; j++){
        ETA_block(i,j) = ETA(j,Class(i),block);
        if(G_version == 1){
          Gs(i,j,block) = ETA_block(i,j);
        }
        
        if(G_version==3){
          Gs(i,j,block)= (t_star+1.)/(T);
        }
        
      }
      if(G_version == 2){
        Gs.slice(block).row(i) = G2vec_efficient(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),
                 (Test_versions(i)-1),test_order,t_star).t();
      }
    }
    
    for(unsigned int j = 0; j < Jt; j++){
      eta_j = ETA_block.col(j);
      
      // sample the response model parameters
      us = R::runif(0, 1);
      ug = R::runif(0, 1);
      // get posterior a, b for sj and gj
      as = arma::sum(eta_j % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      bs = arma::sum(eta_j % Res_block.col(j)) + 1;
      ag = arma::sum((arma::ones<arma::vec>(N)-eta_j) % Res_block.col(j)) + 1;
      bg = arma::sum((arma::ones<arma::vec>(N)-eta_j) % (arma::ones<arma::vec>(N) - Res_block.col(j))) + 1;
      // update g based on s on previous iteration
      pg = R::pbeta(1.0 - itempars(j,0,block), ag, bg, 1, 0);
      itempars(j,1,block) = R::qbeta(ug*pg, ag, bg, 1, 0);
      // update s based on current g
      ps = R::pbeta(1.0 - itempars(j,1,block), as, bs, 1, 0);
      itempars(j,0,block) = R::qbeta(us*ps, as, bs, 1, 0);
      
      // sample the RT model parameters
      // scl_tmp: (log(L_ij) + tau_i + phi * G_ij -gamma_j)^2
      // mu_tmp: log(L_it) + tau_i + phi * G_ij
      arma::vec scl_tmp(N);
      arma::vec mu_tmp(N);
      for(unsigned int i = 0; i < N; i++){
        tau_i = taus(i);
        scl_tmp(i) = pow((log(RT_block(i,j))+tau_i+phi*Gs(i,j,block)-RT_itempars(j,1,block)),2);
        mu_tmp(i) = log(RT_block(i,j))+tau_i+phi*Gs(i,j,block);
      }
      // update alpha_j based on previous gamma_j
      a_alpha = a_alpha0 + N/2;
      // note: the derivation we have corresponds to the rate of gamma, need to take recip for scl
      scl_alpha = 1./(rate_alpha0 + 1./2 * arma::sum(scl_tmp));
      alpha_sqr = R::rgamma(a_alpha,scl_alpha);
      RT_itempars(j,0,block) = sqrt(alpha_sqr);
      // update gamma_j based on current alpha_j
      mu_gamma = (alpha_sqr*arma::sum(mu_tmp))/(N*alpha_sqr + 1);
      sd_gamma = sqrt(1./(N*alpha_sqr + 1));
      RT_itempars(j,1,block) = R::rnorm(mu_gamma, sd_gamma);
    }
  }
  
  
  // update phi
  double num = 0;
  double denom = 0;
  unsigned int test_version_i, block_it;
  for(unsigned int i = 0; i<N; i++){
    tau_i = taus(i);
    test_version_i = Test_versions(i)-1;
    for(unsigned int t = 0; t<T; t++){
      block_it = test_order(test_version_i,t)-1;
      for(unsigned int j = 0; j<Jt; j++){
        // add practice*a of (i,j,block) to num and denom of phi
        num += pow(RT_itempars(j,0,block_it),2) * Gs(i,j,block_it)*(log(latency(i,j,t))-RT_itempars(j,1,block_it)+tau_i);
        denom += pow(RT_itempars(j,0,block_it),2)* pow(Gs(i,j,block_it),2);
      }
    }
  }
  
  double mu_phi, sigma_phi;
  mu_phi = -num/(denom+1.);
  sigma_phi = sqrt(1./(denom+1.));
  phi_vec(0) = R::rnorm(mu_phi,sigma_phi);
  return Rcpp::List::create(Rcpp::Named("accept_theta",accept_theta),
                            Rcpp::Named("accept_lambdas",accept_lambdas)
  );
}


// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_HO_RT_joint(const arma::cube& Response, const arma::cube& Latency,
                                  const arma::cube& Qs, const Rcpp::List Q_examinee,
                                  const arma::mat& test_order, const arma::vec& Test_versions, int G_version,
                                  const double sig_theta_propose, const arma::vec deltas_propose,
                                  const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  // initialize parameters
  arma::vec lambdas_init(3);
  lambdas_init(0) = R::rnorm(0,1);
  lambdas_init(1) = R::runif(0,1);
  lambdas_init(2) = R::runif(0,1);
  
  arma::mat Sig_init = arma::eye<arma::mat>(2,2);
  arma::mat thetatau_init(N,2);
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  for(unsigned int i = 0; i < N; i++){
    thetatau_init.row(i) = rmvnrm(arma::zeros<arma::vec>(2), Sig_init).t();
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  arma::vec thetas_init = thetatau_init.col(0);
  arma::vec taus_init = thetatau_init.col(1);
  arma::cube Alphas_init = simulate_alphas_HO_joint(lambdas_init,thetatau_init.col(0),Alphas_0_init,
                                                    Q_examinee, T, Jt);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec phi_init(1);
  phi_init(0) = R::runif(0,1);
  
  arma::cube itempars_init = .3 * arma::randu<arma::cube>(Jt,2,T);
  itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) =
    itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) % (1.-itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)));
  arma::cube RT_itempars_init(Jt,2,T);
  RT_itempars_init.subcube(0,0,0,(Jt-1),0,(T-1)) = 2.+2.*arma::randu<arma::cube>(Jt,1,T);
  RT_itempars_init.subcube(0,1,0,(Jt-1),1,(T-1)) = arma::randn<arma::cube>(Jt,1,T)*.5+3.45;
  
  
  
  arma::mat S = arma::eye<arma::mat>(2,2);
  double p = 3.;
  //
  // Create objects for storage
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat ss(J,(chain_length-burn_in));
  arma::mat gs(J,(chain_length-burn_in));
  arma::mat RT_as(J,(chain_length-burn_in));
  arma::mat RT_gammas(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat thetas(N,(chain_length-burn_in));
  arma::mat taus(N,(chain_length-burn_in));
  arma::mat lambdas(3,(chain_length-burn_in));
  arma::vec phis(chain_length-burn_in);
  arma::cube Sigs(2,2,(chain_length-burn_in));
  double accept_rate_theta = 0;
  arma::vec accept_rate_lambdas = arma::zeros<arma::vec>(3);
  // arma::cube time_pp(N,T,(chain_length-burn_in));
  // arma::cube res_pp(N,T,(chain_length-burn_in));
  // arma::vec Deviance(chain_length-burn_in);
  // arma::vec Deviance_DINA(chain_length-burn_in);
  // arma::vec Deviance_RT(chain_length-burn_in);
  // arma::vec Deviance_tran(chain_length-burn_in);
  // arma::vec Deviance_joint(chain_length-burn_in);
  
  double tmburn;
  // double deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for(unsigned int tt = 0; tt < chain_length; tt ++){
    Rcpp::List tmp = parm_update_HO_RT_joint(N, Jt, K, T, Alphas_init, pi_init, lambdas_init,
                                             thetas_init, Latency, RT_itempars_init, taus_init,
                                             phi_init, Sig_init, Response, itempars_init, Qs,
                                             Q_examinee, test_order, Test_versions, G_version,
                                             sig_theta_propose, S, p, deltas_propose, 1., 1.);
    if(tt>=burn_in){
      tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
        ss.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(0);
        gs.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = itempars_init.slice(t).col(1);
        RT_as.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(0);
        RT_gammas.rows(Jt*t, (Jt*(t + 1) - 1)).col(tmburn) = RT_itempars_init.slice(t).col(1);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      thetas.col(tmburn) = thetas_init;
      taus.col(tmburn) = taus_init;
      lambdas.col(tmburn) = lambdas_init;
      phis(tmburn) = phi_init(0);
      Sigs.slice(tmburn) = Sig_init;
      
      accept_theta_vec = Rcpp::as<arma::vec>(tmp[0]);
      accept_lambdas_vec = Rcpp::as<arma::vec>(tmp[1]);
      m_accept_theta = arma::mean(accept_theta_vec);
      accept_rate_theta = (accept_rate_theta*tmburn+m_accept_theta)/(tmburn+1.);
      accept_rate_lambdas = (accept_rate_lambdas*tmburn + accept_lambdas_vec)/(tmburn+1.);
      
      // ETA = Rcpp::as<arma::cube>(tmp[2]);
      // J_incidence = Rcpp::as<arma::cube>(tmp[3]);
      
      // //Deviance at this iteration (We use the joint distribution)
      // 
      // double tran=0, response=0, time=0,joint=0;
      // arma::vec G_it = arma::zeros<arma::vec>(Jt);
      // arma::vec vv = bijectionvector(K);
      // arma::vec thetatau(2);
      // 
      // for (unsigned int i = 0; i < N; i++) {
      //   int test_version_i = Test_versions(i) - 1;
      //   
      //    // the joint distribution of theta and tau
      //     thetatau(0)=thetas_init(i);
      //     thetatau(1)=taus_init(i);
      //     
      //     joint +=  std::log (dmvnrm(thetatau,arma::zeros<arma::vec>(2),Sig_init));
      //  
      //   
      //   for (unsigned int t = 0; t < T; t++) {
      //     
      //     // The transition model part
      //     if (t < (T - 1)) {
      //       tran += std::log(pTran(Alphas_init.slice(t).row(i).t(),
      //                              Alphas_init.slice(t + 1).row(i).t(),
      //                              lambdas_init, thetas_init(i), Q_examinee[i], Jt, t));
      //     }
      //     
      //     // The log likelihood from response time model
      //     int test_block_it = test_order(test_version_i, t) - 1;
      //     double class_it = arma::dot(Alphas_init.slice(t).row(i).t(), vv);
      //     if (G_version == 1) {
      //       G_it = ETA.slice(test_block_it).col(class_it);
      //     }
      //     if (G_version == 2) {
      //       G_it = G2vec_efficient(ETA, J_incidence, Alphas_init.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
      //                              test_order, t);
      //     }
      //     
      //     if(G_version==3){
      //       
      //       G_it = arma::ones<arma::vec>(Jt);
      //       arma::vec y(Jt);y.fill((t+1.)/T);
      //       G_it =G_it % y;
      //     }
      //     
      //     time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), RT_itempars_init.slice(test_block_it),
      //                           taus_init(i), phis(tmburn)));
      //     
      //     // The loglikelihood from the DINA
      //     
      //     response += std::log(pYit(ETA.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), itempars_init.slice(test_block_it)));
      //     if (t == 0) {
      //       response += std::log(pi_init(class_it));
      //     }
      //     
      //   }
      //   
      //   
      // }
      // 
      // // Calculate the deviance in this iteration 
      // 
      // deviance = response + time + tran+joint;
      // Deviance(tmburn)=-2*deviance;
      // Deviance_DINA(tmburn)=-2*response;
      // Deviance_RT(tmburn)=-2*(time);
      // Deviance_tran(tmburn)=-2*tran;
      // Deviance_joint(tmburn)=-2*joint;
      
      //         // ppp
      // 		  // Generate response time
      // 		  arma::cube L_model = sim_RT(Alphas_init, RT_itempars_init, Qs, taus_init, phi_init(0), ETA, G_version,
      // 			  test_order, Test_versions);
      // 
      // 		  //Generate the response from DINA model
      // 		  // store the total response time for each block, the total score of each person on each block;
      // 		  arma::cube Res_model = simDINA(Alphas_init, itempars_init, ETA, test_order, Test_versions);
      // 
      // 		  arma::mat L_it_model(N, T), DINA_it_model(N, T);
      // 		  for (unsigned int t = 0; t < T; t++) {
      // 			  L_it_model.col(t) = arma::sum(L_model.slice(t), 1);
      // 			  DINA_it_model.col(t) = arma::sum(Res_model.slice(t), 1);
      // 		  }
      // 
      // 		  time_pp.slice(tmburn) = L_it_model;
      // 		  res_pp.slice(tmburn) = DINA_it_model;
      
      
      
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",ss),
                            Rcpp::Named("gs",gs),
                            Rcpp::Named("as",RT_as),
                            Rcpp::Named("gammas",RT_gammas),
                            Rcpp::Named("pis", pis),
                            Rcpp::Named("thetas",thetas),
                            Rcpp::Named("taus",taus),
                            Rcpp::Named("lambdas",lambdas),
                            Rcpp::Named("phis",phis),
                            Rcpp::Named("Sigs",Sigs),
                            Rcpp::Named("accept_rate_theta",accept_rate_theta),
                            Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas)
                              // Rcpp::Named("time_pp", time_pp),
                              // Rcpp::Named("res_pp", res_pp),
                              // Rcpp::Named("Deviance",Deviance),
                              // Rcpp::Named("D_DINA", Deviance_DINA),
                              // Rcpp::Named("D_RT", Deviance_RT),
                              // Rcpp::Named("D_tran",Deviance_tran),
                              // Rcpp::Named("D_joint",Deviance_joint)
  );
}



//[[Rcpp::export]]
void parm_update_rRUM(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                      arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, 
                      arma::cube& r_stars, arma::mat& pi_stars, const arma::cube Qs, 
                      const arma::cube& responses, arma::cube& X_ijk, arma::cube& Smats, arma::cube& Gmats,
                      const arma::mat& test_order,const arma::vec& Test_versions, const arma::vec& dirich_prior){
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj,prodXijk,pi_ijk,aik,u,compare;
  double pi_ik,aik_nmrtr_k,aik_dnmntr_k,c_aik_1,c_aik_0,ptranspost_1,ptranspost_0,ptransprev_1,ptransprev_0;
  arma::vec aik_nmrtr(K);
  arma::vec aik_dnmntr(K);
  double D_bar = 0;
  arma::mat Classes(N,(T));
  
  // update X
  for(unsigned int i=1;i<N;i++){
    unsigned int test_version_it = Test_versions(i)-1;
    for(unsigned int t=0; t<(T); t++){
      unsigned int block = test_order(test_version_it,t)-1;
      arma::vec pi_star_it = pi_stars.col(block);
      arma::mat r_star_it = r_stars.slice(block);
      arma::mat Q_it = Qs.slice(block);
      
      arma::vec alpha_i =(alphas.slice(t).row(i)).t();
      arma::vec Yi =(responses.slice(t).row(i)).t();
      arma::vec ui = arma::randu<arma::vec>(K);
      aik_nmrtr    = arma::ones<arma::vec>(K);
      aik_dnmntr   = arma::ones<arma::vec>(K);
      
      for(unsigned int j=0;j<Jt;j++){
        double Yij = Yi(j);
        // Note that the Xijk cube is indexed from j = 1 to Jk*(T)
        unsigned int j_star = block*Jt+j;
        arma::vec Xij = X_ijk.tube(i,j_star); 
        arma::uvec task_ij = find(Q_it.row(j) == 1);
        
        for(unsigned int k = 0;k<task_ij.n_elem ;k++){
          kj = task_ij(k);
          aik = alpha_i(kj);
          Xij(kj) = 1;
          prodXijk = prod(Xij(task_ij));
          u = R::runif(0.0,1.0);
          pi_ijk = (1.0-prodXijk)*(aik*(1.0-Smats(j,kj,block)) + (1.0-aik)*Gmats(j,kj,block) );
          compare=(pi_ijk>u);
          Xij(kj)=(1.0-Yij)*compare + Yij;
          
          aik_nmrtr(kj) = ( Xij(kj)*(1.0-Smats(j,kj,block)) + (1.0-Xij(kj))*Smats(j,kj,block) )*aik_nmrtr(kj);
          aik_dnmntr(kj) = ( Xij(kj)*Gmats(j,kj,block) + (1.0-Xij(kj))*(1.0-Gmats(j,kj,block)) )*aik_dnmntr(kj);
        }
        X_ijk.tube(i,j_star) = Xij;
      }
      // Rcpp::Rcout<<aik_nmrtr<<std::endl;
      // Rcpp::Rcout<<aik_dnmntr<<std::endl;
      
      //Update alpha_ikt
      for(unsigned int k=0;k<K;k++){
        arma::vec alpha_i_1 = alpha_i;
        alpha_i_1(k) = 1.0;
        c_aik_1 = (arma::conv_to< double >::from( alpha_i_1.t()*bijectionvector(K) ));
        arma::vec alpha_i_0 = alpha_i;
        alpha_i_0(k) = 0.0;
        c_aik_0 = (arma::conv_to< double >::from( alpha_i_0.t()*bijectionvector(K) ));
        // Rcpp::Rcout<<alpha_i_1<<std::endl;
        // Rcpp::Rcout<<alpha_i_0<<std::endl;
        
        // initial time point
        if(t == 0){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          // Rcpp::Rcout<<alpha_post<<std::endl;
          // 
          // Rcpp::Rcout<<ptranspost_1<<std::endl;
          // Rcpp::Rcout<<ptranspost_0<<std::endl;
          aik_nmrtr_k = aik_nmrtr(k)*pi(c_aik_1)*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*pi(c_aik_0)*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        
        // middle points
        if(t > 0 && t < (T-1)){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        // last time point
        if(t == (T-1)){
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
      }
      alphas.slice(t).row(i) = alpha_i.t();
      // Get DIC
      D_bar += log(pYit_rRUM(alpha_i,Yi,pi_star_it,r_star_it,Q_it));
    }
  }
  
  for(unsigned int t = 0; t<(T); t++){
    Classes.col(t) = alphas.slice(t) * bijectionvector(K);
  }
  
  arma::vec CLASS_0 = alphas.slice(0) * bijectionvector(K);
  
  // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +dirich_prior;
  pi = rDirichlet(deltatilde);
  
  // update item parameters
  //update Smat and Gmat
  double pg,ps,ug,us,gjk,sjk;
  
  for(unsigned int j_star=0;j_star<(Jt*(T));j_star++){
    unsigned int test_version_j = floor(j_star/Jt);
    unsigned int j = j_star % Jt;
    arma::uvec task_ij = find(Qs.slice(test_version_j).row(j) == 1);
    arma::mat Xj = X_ijk.tube(0,j_star,N-1,j_star);
    double pistar_temp =1.0;
    
    arma::mat alpha(N,K);
    for(unsigned int i = 0; i<N; i++){
      unsigned int t_star = arma::conv_to<unsigned int>::from(arma::find(test_order.row(Test_versions(i)-1)==(test_version_j+1)));
      alpha.row(i) = alphas.slice(t_star).row(i);
    }
    
    for(unsigned int k = 0;k<task_ij.n_elem ;k++){
      kj = task_ij(k);
      arma::vec Xjk = Xj.col(kj);
      arma::vec ak = alpha.col(kj);
      
      double Sumalphak =  (arma::conv_to< double >::from(ak.t() * ak));
      double SumXjk = (arma::conv_to< double >::from(Xjk.t() * Xjk));
      double SumXjkalphak = (arma::conv_to< double >::from(Xjk.t() * ak));
      double bsk = SumXjkalphak ;
      double ask = Sumalphak - SumXjkalphak ;
      double agk = SumXjk - SumXjkalphak ;
      double bgk = N - SumXjk - Sumalphak + SumXjkalphak ;
      ug = R::runif(0.0,1.0);
      us = R::runif(0.0,1.0);
      
      //draw g conditoned upon s_t-1
      pg = R::pbeta(1.0-Smats(j,kj,test_version_j),agk+1.0,bgk+1.0,1,0);
      gjk = R::qbeta(ug*pg,agk+1.0,bgk+1.0,1,0);
      //draw s conditoned upon g
      ps = R::pbeta(1.0-gjk,ask+1.0,bsk+1.0,1,0);
      sjk = R::qbeta(us*ps,ask+1.0,bsk+1.0,1,0);
      
      Gmats(j,kj,test_version_j) = gjk;
      Smats(j,kj,test_version_j) = sjk;
      
      r_stars(j,kj,test_version_j) = gjk/(1.0 - sjk);//compute rstarjk
      pistar_temp = (1.0-sjk)*pistar_temp;//compute pistarj
    }
    pi_stars(j,test_version_j) = pistar_temp;
  }
  
  // Update transition probabilities
  for(unsigned int k = 0; k<K; k++){
    double a_tau, b_tau;
    a_tau = b_tau = 1;
    arma::uvec prereqs = arma::find(R.row(k)==1);
    if(prereqs.n_elem==0){
      for(unsigned int t =0; t<(T-1); t++){
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    if(prereqs.n_elem>0){
      for(unsigned int t=0; t<(T-1); t++){
        arma::mat a_prereqs = alphas.slice(t+1).cols(prereqs);
        arma::vec a_preq_1 = arma::prod(a_prereqs,1);
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0 && a_preq_1 == 1);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    taus(k) = R::rbeta((a_tau+1), (b_tau+1));
  }
  
}


// [[Rcpp::export]]
Rcpp::List Gibbs_rRUM_indept(const arma::cube& Response, const arma::cube& Qs, const arma::mat& R,
                             const arma::mat& test_order, const arma::vec& Test_versions,
                             const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Jt*T;
  
  // initialize parameters
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec dirich_prior = arma::ones<arma::vec>(nClass);
  for(unsigned int cc = 0; cc < nClass; cc++){
    arma::vec alpha_cc = inv_bijectionvector(K,cc);
    for(unsigned int k = 0; k<K; k++){
      arma::uvec prereqs = arma::find(R.row(k)==1);
      if(prereqs.n_elem==0){
        if(alpha_cc(k)==1 && arma::prod(alpha_cc(prereqs))==0){
          pi_init(cc) = 0;
          dirich_prior(cc) = 0;
        }
      }
    }
  }
  
  pi_init = pi_init/arma::sum(pi_init);
  
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  for(unsigned int i = 0; i < N; i++){
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  
  arma::vec taus_init = .5*arma::ones<arma::vec>(K);
  
  arma::cube Alphas_init = simulate_alphas_indept(taus_init,Alphas_0_init,T,R);
  
  arma::cube r_stars_init = .5 + .2*arma::randu<arma::cube>(Jt,K,T);
  arma::mat pi_stars_init = .7 + .2*arma::randu<arma::mat>(Jt,T);
  
  arma::cube Smats_init = arma::randu<arma::cube>(Jt,K,T);
  arma::cube Gmats_init = arma::randu<arma::cube>(Jt,K,T) % (1-Smats_init);
  
  arma::cube X = arma::ones<arma::cube>(N,(Jt*T),K);
  
  
  // Create objects for storage 
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::cube r_stars(J,K,(chain_length-burn_in));
  arma::mat pi_stars(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat taus(K,(chain_length-burn_in));
  
  arma::mat Trajectories_mat(N,(K*T));
  arma::vec vv = bijectionvector(K*T);
  
  
  for(unsigned int tt = 0; tt < chain_length; tt++){
    parm_update_rRUM(N,Jt,K,T,Alphas_init,pi_init,taus_init,R,r_stars_init,pi_stars_init,
                     Qs, Response, X, Smats_init, Gmats_init, test_order, Test_versions,
                     dirich_prior);
    
    if(tt>=burn_in){
      unsigned int tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
        r_stars.slice(tmburn).rows(Jt*t,(Jt*(t+1)-1)) = r_stars_init.slice(t);
        pi_stars.rows(Jt*t,(Jt*(t+1)-1)).col(tmburn) = pi_stars_init.col(t);
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      taus.col(tmburn) = taus_init;
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("r_stars",r_stars),
                            Rcpp::Named("pi_stars",pi_stars),
                            Rcpp::Named("pis",pis),
                            Rcpp::Named("taus",taus));
}



//[[Rcpp::export]]
void parm_update_NIDA_indept(const unsigned int N, const unsigned int Jt, const unsigned int K, const unsigned int T,
                             arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, const arma::cube Qs, 
                             const arma::cube& responses, arma::cube& X_ijk, arma::cube& Smats, arma::cube& Gmats,
                             const arma::mat& test_order,const arma::vec& Test_versions, const arma::vec& dirich_prior){
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj,prodXijk,pi_ijk,aik,u,compare;
  double pi_ik,aik_nmrtr_k,aik_dnmntr_k,c_aik_1,c_aik_0,ptranspost_1,ptranspost_0,ptransprev_1,ptransprev_0;
  arma::vec aik_nmrtr(K);
  arma::vec aik_dnmntr(K);
  double D_bar = 0;
  arma::mat Classes(N,(T));
  
  // update X
  for(unsigned int i=1;i<N;i++){
    unsigned int test_version_it = Test_versions(i)-1;
    for(unsigned int t=0; t<(T); t++){
      unsigned int block = test_order(test_version_it,t)-1;
      arma::mat Q_it = Qs.slice(block);
      arma::vec alpha_i =(alphas.slice(t).row(i)).t();
      arma::vec Yi =(responses.slice(t).row(i)).t();
      arma::vec ui = arma::randu<arma::vec>(K);
      aik_nmrtr    = arma::ones<arma::vec>(K);
      aik_dnmntr   = arma::ones<arma::vec>(K);
      
      for(unsigned int j=0;j<Jt;j++){
        double Yij = Yi(j);
        // Note that the Xijk cube is indexed from j = 1 to Jk*(T+1)
        unsigned int j_star = block*Jt+j;
        arma::vec Xij = X_ijk.tube(i,j_star); 
        arma::uvec task_ij = find(Q_it.row(j) == 1);
        
        for(unsigned int k = 0;k<task_ij.n_elem ;k++){
          kj = task_ij(k);
          aik = alpha_i(kj);
          Xij(kj) = 1;
          prodXijk = prod(Xij(task_ij));
          u = R::runif(0.0,1.0);
          pi_ijk = (1.0-prodXijk)*(aik*(1.0-Smats(j,kj,block)) + (1.0-aik)*Gmats(j,kj,block) );
          compare=(pi_ijk>u);
          Xij(kj)=(1.0-Yij)*compare + Yij;
          
          aik_nmrtr(kj) = ( Xij(kj)*(1.0-Smats(j,kj,block)) + (1.0-Xij(kj))*Smats(j,kj,block) )*aik_nmrtr(kj);
          aik_dnmntr(kj) = ( Xij(kj)*Gmats(j,kj,block) + (1.0-Xij(kj))*(1.0-Gmats(j,kj,block)) )*aik_dnmntr(kj);
        }
        X_ijk.tube(i,j_star) = Xij;
      }
      
      //Update alpha_ikt
      for(unsigned int k=0;k<K;k++){
        arma::vec alpha_i_1 = alpha_i;
        alpha_i_1(k) = 1.0;
        c_aik_1 = (arma::conv_to< double >::from( alpha_i_1.t()*bijectionvector(K) ));
        arma::vec alpha_i_0 = alpha_i;
        alpha_i_0(k) = 0.0;
        c_aik_0 = (arma::conv_to< double >::from( alpha_i_0.t()*bijectionvector(K) ));
        // Rcpp::Rcout<<alpha_i_1<<std::endl;
        // Rcpp::Rcout<<alpha_i_0<<std::endl;
        
        // initial time point
        if(t == 0){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          aik_nmrtr_k = aik_nmrtr(k)*pi(c_aik_1)*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*pi(c_aik_0)*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        
        // middle points
        if(t > 0 && t < (T-1)){
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost_1 = pTran_indept(alpha_i_1,alpha_post,taus,R);
          ptranspost_0 = pTran_indept(alpha_i_0,alpha_post,taus,R);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1*ptranspost_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0*ptranspost_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
        // last time point
        if(t == (T-1)){
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev_1 = pTran_indept(alpha_pre,alpha_i_1,taus,R);
          ptransprev_0 = pTran_indept(alpha_pre,alpha_i_0,taus,R);
          
          aik_nmrtr_k = aik_nmrtr(k)*ptransprev_1;
          aik_dnmntr_k = aik_dnmntr(k)*ptransprev_0;
          
          pi_ik = aik_nmrtr_k/(aik_nmrtr_k + aik_dnmntr_k);
          alpha_i(k) = 1.0*(pi_ik > ui(k));    
        }
      }
      alphas.slice(t).row(i) = alpha_i.t();
      
      // Get DIC
      arma::vec Svec = Smats.slice(0).row(0).t();
      arma::vec Gvec = Gmats.slice(0).row(0).t();
      D_bar += log(pYit_NIDA(alpha_i,Yi,Svec,Gvec,Q_it));
    }
  }
  
  for(unsigned int t = 0; t<(T); t++){
    Classes.col(t) = alphas.slice(t) * bijectionvector(K);
  }
  
  arma::vec CLASS_0 = alphas.slice(0) * bijectionvector(K);
  
  // update pi
  arma::uvec class_sum=arma::hist(CLASS_0,arma::linspace<arma::vec>(0,(pow(2,K))-1,(pow(2,K))));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +dirich_prior;
  pi = rDirichlet(deltatilde);
  
  // update item parameters
  //update Smat and Gmat
  double pg,ps,ug,us,gk,sk,bsk,ask,bgk,agk, aikt,Xijk,qijk;
  
  for(unsigned int k = 0;k<K;k++){
    bsk = ask = agk = bgk = 0;
    for(unsigned int t = 0; t<(T); t++){
      for(unsigned int i = 0; i<N; i++){
        aikt = alphas(i,k,t);
        unsigned int test_version_it = Test_versions(i)-1;
        unsigned int block = test_order(test_version_it,t)-1;
        for(unsigned int j = 0; j<Jt; j++){
          unsigned int j_star = block*Jt+j;
          Xijk = X_ijk(i,j_star,k); 
          qijk = Qs(j,k,block);
          ask += aikt*(1-Xijk)*qijk;
          bsk += aikt*Xijk*qijk;
          agk += (1-aikt)*Xijk*qijk;
          bgk += (1-aikt)*(1-Xijk)*qijk;
        }
      }
    }
    ug = R::runif(0.0,1.0);
    us = R::runif(0.0,1.0);
    
    //draw g conditoned upon s_t-1
    pg = R::pbeta(1.0-Smats(0,k,0),agk+1.0,bgk+1.0,1,0);
    gk = R::qbeta(ug*pg,agk+1.0,bgk+1.0,1,0);
    //draw s conditoned upon g
    ps = R::pbeta(1.0-gk,ask+1.0,bsk+1.0,1,0);
    sk = R::qbeta(us*ps,ask+1.0,bsk+1.0,1,0);
    
    Gmats.tube(0,k,(Jt-1),k).fill(gk);
    Smats.tube(0,k,(Jt-1),k).fill(sk);
  }
  
  // Update transition probabilities
  for(unsigned int k = 0; k<K; k++){
    double a_tau, b_tau;
    a_tau = b_tau = 1;
    arma::uvec prereqs = arma::find(R.row(k)==1);
    if(prereqs.n_elem==0){
      for(unsigned int t =0; t<(T-1); t++){
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    if(prereqs.n_elem>0){
      for(unsigned int t=0; t<(T-1); t++){
        arma::mat a_prereqs = alphas.slice(t+1).cols(prereqs);
        arma::vec a_preq_1 = arma::prod(a_prereqs,1);
        arma::uvec subjs = arma::find(alphas.slice(t).col(k)==0 && a_preq_1 == 1);
        arma::vec a_k_new = alphas.slice(t+1).col(k);
        a_tau += arma::sum(a_k_new(subjs));
        b_tau += arma::sum(arma::ones<arma::vec>(subjs.n_elem)-(a_k_new(subjs)));
      }
    }
    taus(k) = R::rbeta((a_tau+1), (b_tau+1));
  }
}


//[[Rcpp::export]]
Rcpp::List Gibbs_NIDA_indept(const arma::cube& Response, const arma::cube& Qs, const arma::mat& R,
                             const arma::mat& test_order, const arma::vec& Test_versions,
                             const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Qs.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Qs.n_cols;
  unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  
  // initialize parameters
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec dirich_prior = arma::ones<arma::vec>(nClass);
  for(unsigned int cc = 0; cc < nClass; cc++){
    arma::vec alpha_cc = inv_bijectionvector(K,cc);
    for(unsigned int k = 0; k<K; k++){
      arma::uvec prereqs = arma::find(R.row(k)==1);
      if(prereqs.n_elem==0){
        if(alpha_cc(k)==1 && arma::prod(alpha_cc(prereqs))==0){
          pi_init(cc) = 0;
          dirich_prior(cc) = 0;
        }
      }
    }
  }
  pi_init = pi_init/arma::sum(pi_init);
  
  arma::mat Alphas_0_init(N,K);
  arma::vec A0vec = arma::randi<arma::vec>(N, arma::distr_param(0,(nClass-1)));
  for(unsigned int i = 0; i < N; i++){
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  
  arma::vec taus_init = .5*arma::ones<arma::vec>(K);
  
  arma::cube Alphas_init = simulate_alphas_indept(taus_init,Alphas_0_init,T,R);
  
  arma::cube r_stars_init = .5 + .2*arma::randu<arma::cube>(Jt,K,T);
  arma::mat pi_stars_init = .7 + .2*arma::randu<arma::mat>(Jt,T);
  
  arma::cube Smats_init = arma::randu<arma::cube>(Jt,K,T);
  arma::cube Gmats_init = arma::randu<arma::cube>(Jt,K,T) % (1-Smats_init);
  
  arma::cube X = arma::ones<arma::cube>(N,(Jt*T),K);
  
  
  // Create objects for storage 
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat Ss(K,(chain_length-burn_in));
  arma::mat Gs(K,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat taus(K,(chain_length-burn_in));
  
  arma::mat Trajectories_mat(N,(K*T));
  arma::vec vv = bijectionvector(K*T);
  
  for(unsigned int tt = 0; tt < chain_length; tt++){
    parm_update_NIDA_indept(N,Jt,K,T,Alphas_init,pi_init,taus_init,R,
                            Qs, Response, X, Smats_init, Gmats_init, test_order, Test_versions,
                            dirich_prior);
    
    if(tt>=burn_in){
      unsigned int tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
        Ss.col(tmburn) = Smats_init.slice(0).row(0).t();
        Gs.col(tmburn) = Gmats_init.slice(0).row(0).t();
      }
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      taus.col(tmburn) = taus_init;
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                            Rcpp::Named("ss",Ss),
                            Rcpp::Named("gs",Gs),
                            Rcpp::Named("pis",pis),
                            Rcpp::Named("taus",taus));
}





// [[Rcpp::export]]
void parm_update_DINA_FOHM(unsigned int N,unsigned int J,unsigned int K,unsigned int nClass,
                           unsigned int nT,const arma::cube& Y,const arma::mat& TP,
                           const arma::mat& ETA,arma::vec& ss,arma::vec& gs,arma::mat& CLASS,
                           arma::vec& pi,arma::mat& Omega){
  arma::vec pt_tm1(nClass);
  double cit,class_itp1,class_itm1,us,ug,pg,ps,gnew,snew,sold;
  arma::mat itempars(J,2);
  itempars.col(0) = ss;
  itempars.col(1) = gs;
  
  //update theta classes over times
  for(unsigned int i=0;i<N;i++){
    
    for(unsigned int t=0;t<nT;t++){
      //***select nonmissing y
      arma::rowvec Yit_temp = Y.subcube(i,0,t,i,J-1,t);
      arma::uvec nomiss_it = arma::find_finite(Yit_temp);
      arma::vec Yit = Yit_temp(nomiss_it);
      
      if(t==0){
        class_itp1 = CLASS(i,t+1);
        pt_tm1 = pi%Omega.col(class_itp1);
        arma::uvec pflag = find(TP.col(class_itp1)==1);
        arma::vec pY(pflag.n_elem);
        
        for(unsigned int g=0;g<pflag.n_elem;g++){
          double cc = pflag(g);
          //***select subset of rows for items
          arma::vec ETA_it_temp = ETA.col(cc);
          arma::vec ETA_it = ETA_it_temp(nomiss_it);
          //***need to select subelements of ss and gs
          pY(g) = pYit_DINA(ETA_it,Yit,itempars.rows(nomiss_it));
        }
        arma::vec numerator = pY % pt_tm1(pflag);
        arma::vec PS = numerator/arma::sum(numerator);
        cit = rmultinomial(PS);
        CLASS(i,t) = pflag(cit);
      }
      
      if(t==nT-1){
        class_itm1 = CLASS(i,t-1);
        pt_tm1 = (Omega.row(class_itm1)).t();
        arma::uvec pflag = find(TP.row(class_itm1)==1);
        arma::vec pY(pflag.n_elem);
        
        for(unsigned int g=0;g<pflag.n_elem;g++){
          double cc = pflag(g);
          //***select subset of rows for items
          arma::vec ETA_it_temp = ETA.col(cc);
          arma::vec ETA_it = ETA_it_temp(nomiss_it);
          //***need to select subelements of ss and gs
          pY(g) = pYit_DINA(ETA_it,Yit,itempars.rows(nomiss_it));
        }
        arma::vec numerator = pY % pt_tm1(pflag);
        arma::vec PS = numerator/arma::sum(numerator);
        cit = rmultinomial(PS);
        CLASS(i,t) = pflag(cit);
      }
      
      if( (t>0) & (t<nT-1) ){
        class_itm1 = CLASS(i,t-1);
        class_itp1 = CLASS(i,t+1);
        
        if(class_itm1==class_itp1 ){
          CLASS(i,t) = class_itm1;
        }
        if(class_itm1!=class_itp1 ){
          arma::vec c_temp = (TP.row(class_itm1)).t()%TP.col(class_itp1);
          arma::uvec pflag = find(c_temp==1);
          pt_tm1 = (Omega.row(class_itm1)).t()%Omega.col(class_itp1);
          arma::vec pY(pflag.n_elem);
          
          for(unsigned int g=0;g<pflag.n_elem;g++){
            double cc = pflag(g);
            //***select subset of rows for items
            arma::vec ETA_it_temp = ETA.col(cc);
            arma::vec ETA_it = ETA_it_temp(nomiss_it);
            //***need to select subelements of ss and gs
            pY(g) = pYit_DINA(ETA_it,Yit,itempars.rows(nomiss_it));
          }
          arma::vec numerator = pY % pt_tm1(pflag);
          arma::vec PS = numerator/arma::sum(numerator);
          cit = rmultinomial(PS);
          CLASS(i,t) = pflag(cit);
        }
      }
    }
  }
  //update pi
  arma::uvec class_sum=arma::hist(CLASS.col(0),arma::linspace<arma::vec>(0,nClass-1,nClass));
  arma::vec deltatilde = arma::conv_to< arma::vec >::from(class_sum) +1.;
  pi = rDirichlet(deltatilde);
  
  //update Omega
  arma::mat tran_sum = arma::zeros<arma::mat>(nClass,nClass);
  for(unsigned int t=0;t<nT-1;t++){
    tran_sum = crosstab(CLASS.col(t),CLASS.col(t+1),TP,nClass,nClass) + tran_sum;
  }
  for(unsigned int cc=0;cc<nClass-1;cc++){
    arma::uvec class_ps = find(TP.row(cc)==1);
    arma::vec temp_mat = (tran_sum.row(cc)).t();
    arma::vec delta_tilde = temp_mat(class_ps) +1.;
    arma::vec w_c = rDirichlet(delta_tilde);
    
    for(unsigned int h=0;h<w_c.n_elem;h++){
      Omega(cc,class_ps(h)) = w_c(h);
    }
  }
  
  //update s,g
  arma::mat Ycrosstabmat = arma::ones<arma::mat>(nClass,2);
  for(unsigned int j=0;j<J;j++){
    arma::mat Yeta_sum = arma::zeros<arma::mat>(nClass,2);
    for(unsigned int t=0;t<nT;t++){
      //select nonmissing elements for each item
      arma::vec Yjt_temp = Y.subcube(0,j,t,N-1,j,t);
      arma::uvec nomiss_jt = arma::find_finite(Yjt_temp);
      arma::vec Yjt = Yjt_temp(nomiss_jt);
      //select classes for individuals with nonmissing yij
      arma::vec CLASS_t = CLASS.col(t);
      Yeta_sum = crosstab(CLASS_t(nomiss_jt),Yjt,Ycrosstabmat,nClass,2)+Yeta_sum;
    }
    arma::uvec ETAj1 = find(ETA.row(j)==1);
    arma::mat Yeta_sum_1 = Yeta_sum.rows(ETAj1);
    arma::rowvec ab_s = arma::sum(Yeta_sum_1);
    arma::uvec ETAj0 = find(ETA.row(j)==0);
    arma::mat Yeta_sum_0 = Yeta_sum.rows(ETAj0);
    arma::rowvec ab_g = arma::sum(Yeta_sum_0);
    
    //sample s and g as linearly truncated bivariate beta
    us=R::runif(0,1);
    ug=R::runif(0,1);
    sold = ss(j);
    //draw g conditoned upon s_t-1
    pg = R::pbeta(1.0-sold,ab_g(1)+1.,ab_g(0)+1.,1,0);
    gnew = R::qbeta(ug*pg,ab_g(1)+1.,ab_g(0)+1.,1,0);
    //draw s conditoned upon g
    ps = R::pbeta(1.0-gnew,ab_s(0)+1.,ab_s(1)+1.,1,0);
    snew = R::qbeta(us*ps,ab_s(0)+1.,ab_s(1)+1.,1,0);
    
    gs(j) = gnew;
    ss(j) = snew;
  }
}




// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_FOHM(const arma::cube& Y,const arma::mat& Q,
                           unsigned int burnin,unsigned int chain_length){
  unsigned int N = Y.n_rows;
  unsigned int J = Y.n_cols;
  unsigned int nT = Y.n_slices;
  unsigned int K = Q.n_cols;
  unsigned int C = pow(2,K);
  unsigned int chain_m_burn = chain_length-burnin;
  unsigned int tmburn;
  
  arma::vec vv = bijectionvector(K);
  arma::mat ETA = ETAmat(K,J,Q);
  arma::mat TP = TPmatFree(K);
  arma::vec vvp = bijectionvector(K*nT);
  
  //Savinging output
  arma::mat SS(J,chain_m_burn);
  arma::mat GS(J,chain_m_burn);
  arma::mat PIs(C,chain_m_burn);
  arma::cube OMEGAS(C,C,chain_m_burn);
  // arma::cube CLASStotal(N,nT,chain_m_burn);
  arma::mat Trajectories(N,(chain_m_burn));
  arma::mat Trajectories_mat(N,(K*nT));
  
  //need to initialize, alphas, X,ss, gs,pis 
  arma::mat Omega = rOmegaFree(TP);
  arma::vec class0 = arma::randi<arma::vec>(N,arma::distr_param(0,C-1));
  arma::mat CLASS=rAlpha(Omega,N,nT,class0);
  arma::vec ss = arma::randu<arma::vec>(J);
  arma::vec gs = (arma::ones<arma::vec>(J) - ss)%arma::randu<arma::vec>(J);
  arma::vec delta0 = arma::ones<arma::vec>(C);
  arma::vec pis = rDirichlet(delta0);
  
  //Start Markov chain
  for(unsigned int t = 0; t < chain_length; t++){
    parm_update_DINA_FOHM(N,J,K,C,nT,Y,TP,ETA,ss,gs,CLASS,pis,Omega);
    
    if(t>=burnin){
      tmburn = t-burnin;
      //update parameter value via pointer. save classes and PIs
      SS.col(tmburn)       = ss;
      GS.col(tmburn)       = gs;
      PIs.col(tmburn)      = pis;
      OMEGAS.slice(tmburn) = Omega;
      for(unsigned int i = 0; i<N; i++){
        for(unsigned int tt = 0; tt < nT; tt++){
          Trajectories_mat.cols(K*tt,(K*(tt+1)-1)).row(i) = inv_bijectionvector(K,CLASS(i,tt)).t();
        }
      }
      Trajectories.col(tmburn) = Trajectories_mat * vvp;
    }
    
    if(t%1000==0){
      Rcpp::Rcout<<t<<std::endl;
    }
    
    
  }
  return Rcpp::List::create(Rcpp::Named("ss",SS),
                            Rcpp::Named("gs",GS),
                            Rcpp::Named("pis",PIs),
                            Rcpp::Named("omegas",OMEGAS),
                            Rcpp::Named("trajectories",Trajectories)
  );
}



//' @title Gibbs sampler for learning models
//' @description Runs MCMC to estimate parameters of any of the listed learning models. 
//' @param Response_list A \code{list} of dichotomous item responses. t-th element is an N-by-Jt matrix of responses at time t.
//' @param Q_list A \code{list} of Q-matrices. b-th element is a Jt-by-K Q-matrix for items in block b. 
//' @param model A \code{charactor} of the type of model fitted with the MCMC sampler, possible selections are 
//' "DINA_HO": Higher-Order Hidden Markov Diagnostic Classification Model with DINA responses;
//' "DINA_HO_RT_joint": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and joint modeling of latent
//' speed and learning ability; 
//' "DINA_HO_RT_sep": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and separate modeling of latent
//' speed and learning ability; 
//' "rRUM_indept": Simple independent transition probability model with rRUM responses
//' "NIDA_indept": Simple independent transition probability model with NIDA responses
//' "DINA_FOHM": First Order Hidden Markov model with DINA responses
//' @param test_order A \code{matrix} of the order of item blocks for each test version.
//' @param Test_versions A \code{vector} of the test version of each learner.
//' @param chain_length An \code{int} of the MCMC chain length.
//' @param burn_in An \code{int} of the MCMC burn-in chain length.
//' @param Q_examinee Optional. A \code{list} of the Q matrix for each learner. i-th element is a J-by-K Q-matrix for all items learner i was administered.
//' @param Latency_list Optional. A \code{list} of the response times. t-th element is an N-by-Jt matrix of response times at time t.
//' @param G_version Optional. An \code{int} of the type of covariate for increased fluency (1: G is dichotomous depending on whether all skills required for
//' current item are mastered; 2: G cumulates practice effect on previous items using mastered skills; 3: G is a time block effect invariant across 
//' subjects with different attribute trajectories)
//' @param theta_propose Optional. A \code{scalar} for the standard deviation of theta's proposal distribution in the MH sampling step.
//' @param deltas_propose Optional. A \code{vector} for the band widths of each lambda's proposal distribution in the MH sampling step.
//' @param R Optional. A reachability \code{matrix} for the hierarchical relationship between attributes. 
//' @return A \code{list} of parameter samples and Metropolis-Hastings acceptance rates (if applicable).
//' @author Susu Zhang
//' @examples
//' \donttest{
//' output_FOHM = MCMC_learning(Y_real_list,Q_list,"DINA_FOHM",test_order,Test_versions,10000,5000)
//' }
//' @export
// [[Rcpp::export]]
Rcpp::List MCMC_learning(const Rcpp::List Response_list, const Rcpp::List Q_list, 
                         const std::string model, const arma::mat& test_order, const arma::vec& Test_versions,
                         const unsigned int chain_length, const unsigned int burn_in,
                         const Rcpp::Nullable<Rcpp::List> Q_examinee=R_NilValue,
                         const Rcpp::Nullable<Rcpp::List> Latency_list = R_NilValue, const int G_version = NA_INTEGER,
                         const double theta_propose = 0., const Rcpp::Nullable<Rcpp::NumericVector> deltas_propose = R_NilValue,
                         const Rcpp::Nullable<Rcpp::NumericMatrix> R = R_NilValue){
  Rcpp::List output;
  unsigned int T = test_order.n_rows;
  arma::mat temp = Rcpp::as<arma::mat>(Q_list[0]);
  unsigned int Jt = temp.n_rows;
  unsigned int K = temp.n_cols;
  unsigned int N = Test_versions.n_elem;
  arma::cube Response(N,Jt,T);
  arma::cube Latency(N,Jt,T);
  arma::cube Qs(Jt,K,T);
  for(unsigned int t = 0; t<T; t++){
    Response.slice(t) = Rcpp::as<arma::mat>(Response_list[t]);
    Qs.slice(t) = Rcpp::as<arma::mat>(Q_list[t]);
    if(Latency_list.isNotNull()){
      Rcpp::List tmp = Rcpp::as<Rcpp::List>(Latency_list);
      Latency.slice(t) = Rcpp::as<arma::mat>(tmp[t]);
    }
  }
  if(model == "DINA_HO"){
    
    output = Gibbs_DINA_HO(Response, Qs, Rcpp::as<Rcpp::List>(Q_examinee), test_order, Test_versions, theta_propose, Rcpp::as<arma::vec>(deltas_propose),
                           chain_length, burn_in);
  }
  if(model == "DINA_HO_RT_joint"){
    output = Gibbs_DINA_HO_RT_joint(Response, Latency, Qs, Rcpp::as<Rcpp::List>(Q_examinee), test_order, Test_versions, G_version,
                                    theta_propose, Rcpp::as<arma::vec>(deltas_propose), chain_length, burn_in);
  }
  if(model == "DINA_HO_RT_sep"){
    output = Gibbs_DINA_HO_RT_sep(Response, Latency, Qs, Rcpp::as<Rcpp::List>(Q_examinee), test_order, Test_versions, G_version,
                                  theta_propose, Rcpp::as<arma::vec>(deltas_propose), chain_length, burn_in);
  }
  if(model == "rRUM_indept"){
    output = Gibbs_rRUM_indept(Response, Qs, Rcpp::as<arma::mat>(R),test_order, Test_versions, chain_length, burn_in);
  }
  if(model == "NIDA_indept"){
    output = Gibbs_NIDA_indept(Response, Qs, Rcpp::as<arma::mat>(R), test_order, Test_versions, chain_length, burn_in);
  }
  if(model == "DINA_FOHM"){
    arma::cube Y_miss = resp_miss(Response, test_order, Test_versions);
    unsigned int Jt = Qs.n_rows;
    unsigned int T = Qs.n_slices;
    unsigned int K = Qs.n_cols;
    arma::mat Q_mat(Jt*T, K);
    for(unsigned int t= 0; t<T; t++){
      Q_mat.rows(Jt*t, (Jt*(t+1)-1)) = Qs.slice(t);
    }
    output = Gibbs_DINA_FOHM(Y_miss, Q_mat, burn_in, chain_length);
  }
  
  return(output);
}
