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
Rcpp::List parm_update_HO_g(const arma::cube Design_array, arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                            const arma::cube response, arma::mat& itempars, const Rcpp::List Q_examinee,
                            const double theta_propose, const arma::vec deltas_propose,
                            const arma::mat Q_matrix){
  unsigned int N = Design_array.n_rows;
  unsigned int K = Q_matrix.n_cols;
  unsigned int T = Design_array.n_slices;
  unsigned int J = Q_matrix.n_rows;
  
  arma::mat ETA = ETAmat(K,J,Q_matrix);
  arma::vec CLASS_0(N);
  
  double post_new, post_old;
  double theta_i_new;
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  double theta_i;
  arma::uvec block_it;
  arma::vec response_it;
  arma::vec ETA_cc;
  for(unsigned int i = 0; i<N; i++){
    // int test_version_i = Test_versions(i)-1;
    theta_i = thetas(i);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      block_it = arma::find_finite(Design_array.slice(t).row(i));
      response_it = response.slice(t).row(i).t();
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        ETA_cc = ETA.col(cc);
        likelihood_Y(cc) = pYit_DINA(ETA_cc.elem(block_it), response_it.elem(block_it),itempars.rows(block_it));
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
          ptranspost(cc) = pTran_HO_sep_g(alpha_c,alpha_post,lambdas,theta_i,Q_i,Design_array,t,i);
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
          ptranspost(cc) = pTran_HO_sep_g(alpha_c,alpha_post,lambdas,theta_i,Q_i,Design_array,t,i);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep_g(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Design_array,(t-1),i);
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
          ptransprev(cc) = pTran_HO_sep_g(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Design_array,(t-1),i);
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
      post_old += std::log(pTran_HO_sep_g(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Design_array,(t-1),i));
      post_new += std::log(pTran_HO_sep_g(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i_new,Q_i,Design_array,(t-1),i));
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
        post_old += std::log(pTran_HO_sep_g(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            lambdas, thetas(i), Q_examinee[i], Design_array, t,i));
        post_new += std::log(pTran_HO_sep_g(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            tmp, thetas(i), Q_examinee[i], Design_array, t,i));
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
  arma::mat Res_block(N, J);
  arma::vec Class(N);
  arma::vec eta_j(N);
  arma::mat ETA_block(N,J);
  double t_star;
  arma::vec alpha;
  for(unsigned int i = 0; i < N; i++){
    for(unsigned int j = 0; j < J; j++){
      // find the time point at which i received this block
      t_star = max(arma::find_finite(Design_array.subcube(i,j,0,i,j,T-1)));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.col(j).row(i) = response.slice(t_star).col(j).row(i);
      alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      ETA_block(i,j) = ETA(j,Class(i));
    }
  }
  for(unsigned int j = 0; j < J; j++){
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
    pg = R::pbeta(1.0 - itempars(j,0), ag, bg, 1, 0);
    itempars(j,1) = R::qbeta(ug*pg, ag, bg, 1, 0);
    // update s based on current g
    ps = R::pbeta(1.0 - itempars(j,1), as, bs, 1, 0);
    itempars(j,0) = R::qbeta(us*ps, as, bs, 1, 0);
  }
  
  // return the results
  return Rcpp::List::create(Rcpp::Named("accept_theta",accept_theta),
                            Rcpp::Named("accept_lambdas",accept_lambdas)
  );
}


// [[Rcpp::export]]
Rcpp::List Gibbs_DINA_HO_g(const arma::cube& Response, 
                           const arma::mat& Q_matrix,
                           const arma::cube& Design_array,
                           const double theta_propose,const arma::vec deltas_propose,
                           const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Design_array.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Q_matrix.n_cols;
  // unsigned int Jt = Qs.n_rows;
  unsigned int nClass = pow(2,K);
  unsigned int J = Q_matrix.n_rows;
  Rcpp::List Q_examinee = Q_list_g(Q_matrix, Design_array);
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
    thetas_init(i) = R::rnorm(0, 1);
    Alphas_0_init.row(i) = inv_bijectionvector(K,A0vec(i)).t();
  }
  arma::cube Alphas_init = simulate_alphas_HO_sep_g(lambdas_init,thetas_init,
                                                    Q_matrix,Design_array,Alphas_0_init);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  
  arma::mat itempars_init = .3 * arma::randu<arma::mat>(J,2);
  itempars_init.submat(0,1,J-1,1) = itempars_init.submat(0,1,J-1,1) % (1-itempars_init.submat(0,0,J-1,0));
  
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
  Rcpp::List tmp = parm_update_HO_g(Design_array,Alphas_init,pi_init,lambdas_init,thetas_init,Response,
                                    itempars_init,Q_examinee,theta_propose,deltas_propose,Q_matrix);
  if (tt >= burn_in) {
    tmburn = tt - burn_in;
    for (unsigned int t = 0; t < T; t++) {
      Trajectories_mat.cols(K*t, (K*(t + 1) - 1)) = Alphas_init.slice(t);
    }
    ss.col(tmburn) = itempars_init.col(0);
    gs.col(tmburn) = itempars_init.col(1);
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
  Rcpp::List input_data = Rcpp::List::create(Rcpp::Named("Response",Response),
                                             Rcpp::Named("Q_matrix",Q_matrix),
                                             Rcpp::Named("Q_examinee",Q_examinee),
                                             Rcpp::Named("Design_array",Design_array)
  );
  Rcpp::List res = Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                                      Rcpp::Named("ss",ss),
                                      Rcpp::Named("gs",gs),
                                      Rcpp::Named("pis", pis),
                                      Rcpp::Named("thetas",thetas),
                                      Rcpp::Named("lambdas",lambdas),
                                      Rcpp::Named("accept_rate_theta",accept_rate_theta),
                                      Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas),
                                      
                                      Rcpp::Named("Model", "DINA_HO"),
                                      Rcpp::Named("chain_length", chain_length),
                                      Rcpp::Named("burn_in", burn_in),
                                      
                                      Rcpp::Named("input_data",input_data)
  );
  res.attr("class") = "hmcdm";
  return res;
  
}



// [[Rcpp::export]]
Rcpp::List parm_update_HO_RT_sep_g(const arma::cube Design_array, 
                                 arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                 const arma::cube latency, arma::mat& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::vec& tauvar,
                                 const arma::cube response, arma::mat& itempars, const arma::mat Q_matrix, const Rcpp::List Q_examinee,
                                 const int G_version,
                                 const double theta_propose, const double a_sigma_tau0, const double rate_sigma_tau0, 
                                 const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0){
  unsigned int N = Design_array.n_rows;
  unsigned int J = Design_array.n_cols;
  unsigned int T = Design_array.n_slices;
  unsigned int K = Q_matrix.n_cols;
  double phi = phi_vec(0);
  double tau_sig = tauvar(0);
  arma::mat ETA = ETAmat(K,J,Q_matrix);
  arma::vec CLASS_0(N);
  arma::cube J_incidence = J_incidence_cube_g(Q_matrix,Design_array);
  
  double post_new, post_old;
  arma::vec thetatau_i_old(2);
  arma::vec thetatau_i_new(2);
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  arma::vec accept_tau = arma::zeros<arma::vec>(N);
  
  double theta_i;
  double tau_i;
  arma::uvec block_it;
  arma::vec block_it_vec;
  arma::vec response_it;
  arma::vec latency_it;
  arma::vec G_it;
  unsigned int J_it;
  arma::vec ETA_cc;
  
  for(unsigned int i = 0; i<N; i++){
    theta_i = thetas(i);
    tau_i = taus(i);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      block_it = arma::find_finite(Design_array.slice(t).row(i));
      response_it = response.slice(t).row(i).t();
      latency_it = latency.slice(t).row(i).t();
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        ETA_cc = ETA.col(cc);
        likelihood_Y(cc) = pYit_DINA(ETA_cc.elem(block_it), response_it.elem(block_it),itempars.rows(block_it));
      }
      // likelihood of RT (time dependent)
      arma::vec likelihood_L = arma::ones<arma::vec>(pow(2,K));
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      arma::uvec test_block_itt;
      
      // initial time point
      if(t == 0){
        
        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_sep_g(alpha_c,alpha_post,lambdas,theta_i,Q_i,Design_array,t,i);
          
          // likelihood of RT
          ETA_cc = ETA.col(cc);
          if(G_version !=3){
            if(G_version==1 ){
              G_it = ETA_cc.elem(block_it);
              likelihood_L(cc) = dLit(G_it,latency_it.elem(block_it),RT_itempars.rows(block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =0; tt<T; tt++){
                test_block_itt = arma::find_finite(Design_array.slice(t).row(i));
                G_it = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),tt,
                                       Q_matrix,Design_array,i);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency_it.elem(test_block_itt),
                             RT_itempars.rows(test_block_itt),tau_i,phi);
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
          ptranspost(cc) = pTran_HO_sep_g(alpha_c,alpha_post,lambdas,theta_i,Q_i,Design_array,t,i);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_sep_g(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Design_array,(t-1),i);
          
          // likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              ETA_cc = ETA.col(cc);
              G_it = ETA_cc.elem(block_it);
              likelihood_L(cc) = dLit(G_it,latency_it.elem(block_it),RT_itempars.rows(block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =t; tt<T; tt++){
                test_block_itt = arma::find_finite(Design_array.slice(t).row(i));
                G_it = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),tt,
                                         Q_matrix,Design_array,i);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency_it.elem(test_block_itt),
                             RT_itempars.rows(test_block_itt),tau_i,phi);
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
          ptransprev(cc) = pTran_HO_sep_g(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Design_array,(t-1),i);
          // Likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              ETA_cc = ETA.col(cc);
              G_it = ETA_cc.elem(block_it);
            }
            if(G_version==2){
              G_it = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),t,
                                       Q_matrix,Design_array,i);
            }
            
            
            
            likelihood_L(cc) = dLit(G_it,latency_it.elem(block_it),RT_itempars.rows(block_it),
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
      post_old += std::log(pTran_HO_sep_g(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Design_array,(t-1),i));
      post_new += std::log(pTran_HO_sep_g(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,thetatau_i_new(0),Q_i,Design_array,(t-1),i));
    }
    ratio = exp(post_new - post_old);
    u = R::runif(0,1);
    if(u < ratio){
      thetas(i) = thetatau_i_new(0);
      accept_theta(i) = 1;
    }
    
    // update tau_i, Gbbs, draw the tau_i from the posterior distribution, which is still normal
    
    double num = 0;
    double denom = 0;
    for (unsigned int t = 0; t<T; t++) {
      block_it = arma::find_finite(Design_array.slice(t).row(i));
      block_it_vec = arma::conv_to<arma::vec>::from(block_it);
      J_it = block_it.n_elem;

      if (G_version == 1) {
        double class_it = arma::dot(alphas.slice(t).row(i).t(), vv);
        arma::vec ETA_it = ETA.col(class_it);
        G_it = ETA_it.elem(block_it);
      }
      if (G_version == 2) {
        G_it = G2vec_efficient_g(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), t,
                               Q_matrix, Design_array, i);
      }
      if(G_version==3){
        
        G_it = arma::ones<arma::vec>(J_it);
        arma::vec y(J_it);
        y.fill((t+1.)/T);
        G_it =G_it % y;
      }
      
      
      for (unsigned int j = 0; j<J_it; j++) {
        //add practice*a of (i,j) to num and denom of phi
        block_it_vec.subvec(1,1);
        num += (log(latency(i,block_it_vec(j),t))-RT_itempars(block_it_vec(j), 1)+ phi*G_it(j))*pow(RT_itempars(block_it_vec(j), 0), 2);
        denom += pow(RT_itempars(block_it_vec(j), 0), 2);
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
        post_old += std::log(pTran_HO_sep_g(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas, thetas(i), Q_examinee[i], Design_array, t, i));
        post_new += std::log(pTran_HO_sep_g(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          tmp, thetas(i), Q_examinee[i], Design_array, t, i));
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
  double a_alpha, scl_alpha, mu_gamma, sd_gamma, alpha_sqr, t_star;
  arma::mat Gs(N, J);
  arma::mat ETA_block(N,J);
  arma::vec Class(N);
  arma::vec eta_j(N);
  arma::mat Res_block(N, J);
  arma::mat RT_block(N, J);
  arma::vec alpha;

  for(unsigned int i = 0; i < N; i++){
    tau_i = taus(i);
    // get response, RT, alphas, and Gs for items in this block
    for(unsigned int j = 0; j < J; j++){
      // find the time point at which i received this item
      t_star = max(arma::find_finite(Design_array.subcube(i,j,0,i,j,T-1)));
      alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.col(j).row(i) = response.slice(t_star).col(j).row(i);
      RT_block.col(j).row(i) = latency.slice(t_star).col(j).row(i);
      alpha = alphas.slice(t_star).row(i).t();

      ETA_block(i,j) = ETA(j,Class(i));
      if(G_version == 1){
        Gs(i,j) = ETA_block(i,j);
      }
      if(G_version==3){
        Gs(i,j)= (t_star+1.)/T;
      }
    }
    if(G_version == 2){
      Gs.row(i) = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),
               t_star,Q_matrix,Design_array,i).t();
    }
  }


  for(unsigned int j = 0; j < J; j++){
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
    pg = R::pbeta(1.0 - itempars(j,0), ag, bg, 1, 0);
    itempars(j,1) = R::qbeta(ug*pg, ag, bg, 1, 0);
    // update s based on current g
    ps = R::pbeta(1.0 - itempars(j,1), as, bs, 1, 0);
    itempars(j,0) = R::qbeta(us*ps, as, bs, 1, 0);

    // sample the RT model parameters
    arma::vec scl_tmp(N);
    arma::vec mu_tmp(N);
    for(unsigned int i = 0; i < N; i++){
      tau_i = taus(i);
      scl_tmp(i) = pow((log(RT_block(i,j))+tau_i+phi*Gs(i,j)-RT_itempars(j,1)),2);
      mu_tmp(i) = log(RT_block(i,j))+tau_i+phi*Gs(i,j);
    }
    // update alpha_j based on previous gamma_j
    a_alpha = a_alpha0 + N/2;
    // note: the derivation we have corresponds to the rate of gamma, need to take recip for scl
    scl_alpha = 1./(rate_alpha0 + 1./2 * arma::sum(scl_tmp));
    alpha_sqr = R::rgamma(a_alpha,scl_alpha);
    RT_itempars(j,0) = sqrt(alpha_sqr);
    // update gamma_j based on current alpha_j
    mu_gamma = (alpha_sqr*arma::sum(mu_tmp))/(N*alpha_sqr + 1);
    sd_gamma = sqrt(1./(N*alpha_sqr + 1));
    RT_itempars(j,1) = R::rnorm(mu_gamma, sd_gamma);
  }

  // update phi
  double num = 0;
  double denom = 0;
  for(unsigned int i = 0; i<N; i++){
    tau_i = taus(i);
    for(unsigned int t = 0; t<T; t++){

      for(unsigned int j = 0; j<J; j++){
        // find the time point at which i received this item
        t_star = max(arma::find_finite(Design_array.subcube(i,j,0,i,j,T-1)));
        //add practice*a of (i,j,block) to num and denom of phi
        num += pow(RT_itempars(j,0),2) * Gs(i,j)*(log(latency(i,j,t_star))-RT_itempars(j,1)+tau_i);
        denom += pow(RT_itempars(j,0),2)* pow(Gs(i,j),2);
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
Rcpp::List Gibbs_DINA_HO_RT_sep_g(const arma::cube& Response, const arma::cube& Latency,
                                const arma::mat& Q_matrix, 
                                const arma::cube& Design_array, int G_version,
                                const double theta_propose,const arma::vec deltas_propose,
                                const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Design_array.n_slices;
  unsigned int N = Design_array.n_rows;
  unsigned int K = Q_matrix.n_cols;
  unsigned int J = Q_matrix.n_rows;
  unsigned int nClass = pow(2,K);
  
  Rcpp::List Q_examinee = Q_list_g(Q_matrix, Design_array);
  //std::srand(std::time(0));//use current time as seed for random generator
  
  // initialize parameters
  arma::vec lambdas_init(4),tauvar_init(1);
  lambdas_init(0) = R::rnorm(0,1);
  lambdas_init(1) = R::runif(0,1);
  lambdas_init(2) = R::runif(0,1);
  lambdas_init(3) = R::runif(0, 1);
  
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
  arma::cube Alphas_init = simulate_alphas_HO_sep_g(lambdas_init,thetatau_init.col(0),Q_matrix, Design_array,Alphas_0_init);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec phi_init(1);
  phi_init(0) = R::runif(0,1);

  arma::mat itempars_init = .3 * arma::randu<arma::mat>(J,2);
  itempars_init.submat(0,1,J-1,1) =
    itempars_init.submat(0,1,J-1,1) % (1.-itempars_init.submat(0,0,J-1,0));
  arma::mat RT_itempars_init(J,2);
  RT_itempars_init.submat(0,0,J-1,0) = 2.+2.*arma::randu<arma::mat>(J,1);
  RT_itempars_init.submat(0,1,J-1,1) = arma::randn<arma::mat>(J,1); // why take exponetial part?
  

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
  arma::vec tauvar(chain_length - burn_in);
  double accept_rate_theta = 0;
  arma::vec accept_rate_lambdas = arma::zeros<arma::vec>(4);

  double tmburn;//,deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec,accept_tau_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for (unsigned int tt = 0; tt < chain_length; tt++) {
    Rcpp::List tmp = parm_update_HO_RT_sep_g(Design_array, Alphas_init, pi_init, lambdas_init,
                                           thetas_init, Latency, RT_itempars_init, taus_init,
                                           phi_init, tauvar_init, Response, itempars_init, Q_matrix,
                                           Q_examinee, G_version,
                                           theta_propose, 2.5, 1., deltas_propose, 1., 1.);
    if (tt >= burn_in) {
      
      tmburn = tt - burn_in;
      for (unsigned int t = 0; t < T; t++) {
        Trajectories_mat.cols(K*t, (K*(t + 1) - 1)) = Alphas_init.slice(t);
      }
      ss.col(tmburn) = itempars_init.col(0);
      gs.col(tmburn) = itempars_init.col(1);
      RT_as.col(tmburn) = RT_itempars_init.col(0);
      RT_gammas.col(tmburn) = RT_itempars_init.col(1);
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
    }
    if (tt % 1000 == 0) {
      Rcpp::Rcout << tt << std::endl;
    }
    
  }
  Rcpp::List input_data = Rcpp::List::create(Rcpp::Named("Response",Response),
                                             Rcpp::Named("Latency",Latency),
                                             Rcpp::Named("G_version",G_version),
                                             Rcpp::Named("Q_matrix",Q_matrix),
                                             Rcpp::Named("Q_examinee",Q_examinee),
                                             Rcpp::Named("Design_array",Design_array)
  );
  Rcpp::List res = Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
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
                                      Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas),
                                      
                                      Rcpp::Named("Model", "DINA_HO_RT_sep"),
                                      Rcpp::Named("chain_length",chain_length),
                                      Rcpp::Named("burn_in", burn_in),
                                      
                                      Rcpp::Named("input_data",input_data)
  );
  
  res.attr("class") = "hmcdm";
  return res;
}




// [[Rcpp::export]]
Rcpp::List parm_update_HO_RT_joint_g(const arma::cube& Design_array,
                                   arma::cube& alphas, arma::vec& pi, arma::vec& lambdas, arma::vec& thetas,
                                   const arma::cube latency, arma::mat& RT_itempars, arma::vec& taus, arma::vec& phi_vec, arma::mat& Sig,
                                   const arma::cube response, arma::mat& itempars, const arma::mat Q_matrix, const Rcpp::List Q_examinee,
                                   const int G_version,
                                   const double sig_theta_propose, const arma::mat S, double p,
                                   const arma::vec deltas_propose, const double a_alpha0, const double rate_alpha0
                                       ){
  double N = Design_array.n_rows;
  double J = Design_array.n_cols;
  double T = Design_array.n_slices;
  double K = Q_matrix.n_cols;
  
  double phi = phi_vec(0);
  arma::vec CLASS_0(N);
  arma::mat ETA = ETAmat(K,J,Q_matrix);
  arma::cube J_incidence = J_incidence_cube_g(Q_matrix,Design_array);
  
  double post_new, post_old;
  double theta_new;
  arma::vec thetatau_i_old(2);
  arma::vec thetatau_i_new(2);
  arma::vec vv = bijectionvector(K);
  
  double ratio, u;
  
  arma::vec accept_theta = arma::zeros<arma::vec>(N);
  arma::vec accept_tau = arma::zeros<arma::vec>(N);  
  
  double theta_i;
  double tau_i;
  arma::uvec block_it;
  arma::vec block_it_vec;
  arma::vec response_it;
  arma::vec latency_it;
  arma::vec G_it;
  unsigned int J_it;
  arma::vec ETA_cc;

  for(unsigned int i = 0; i<N; i++){
    theta_i = thetas(i);
    tau_i = taus(i);
    arma::mat Q_i = Q_examinee[i];
    // update alphas
    for(unsigned int t = 0; t< T; t++){
      block_it = arma::find_finite(Design_array.slice(t).row(i));
      response_it = response.slice(t).row(i).t();
      latency_it = latency.slice(t).row(i).t();
      // get likelihood of response
      arma::vec likelihood_Y(pow(2,K));
      for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
        ETA_cc = ETA.col(cc);
        likelihood_Y(cc) = pYit_DINA(ETA_cc.elem(block_it), response_it.elem(block_it),itempars.rows(block_it));
      }
      // likelihood of RT (time dependent)
      arma::vec likelihood_L = arma::ones<arma::vec>(pow(2,K));
      // prob(alpha_it|pre/post)
      arma::vec ptransprev(pow(2,K));
      arma::vec ptranspost(pow(2,K));
      arma::uvec test_block_itt;

      // initial time point
      if(t == 0){

        for(unsigned int cc = 0; cc<(pow(2,K)); cc++){
          // transition probability
          arma::vec alpha_c = inv_bijectionvector(K,cc);
          arma::vec alpha_post = alphas.slice(t+1).row(i).t();
          ptranspost(cc) = pTran_HO_joint_g(alpha_c,alpha_post,lambdas,theta_i,Q_i,Design_array,t,i);

          // likelihood of RT
          if(G_version !=3){
            if(G_version==1 ){
              G_it = ETA_cc.elem(block_it);
              likelihood_L(cc) = dLit(G_it,latency_it.elem(block_it),RT_itempars.rows(block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =0; tt<T; tt++){
                test_block_itt = arma::find_finite(Design_array.slice(t).row(i));
                G_it = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),tt,
                                         Q_matrix,Design_array,i);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency_it.elem(test_block_itt),
                             RT_itempars.rows(test_block_itt),tau_i,phi);
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
          ptranspost(cc) = pTran_HO_joint_g(alpha_c,alpha_post,lambdas,theta_i,Q_i,Design_array,t,i);
          arma::vec alpha_pre = alphas.slice(t-1).row(i).t();
          ptransprev(cc) = pTran_HO_joint_g(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Design_array,(t-1),i);

          // likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              ETA_cc = ETA.col(cc);
              G_it = ETA_cc.elem(block_it);
              likelihood_L(cc) = dLit(G_it,latency_it.elem(block_it),RT_itempars.rows(block_it),
                           tau_i,phi);
            }
            if(G_version==2){
              for(unsigned int tt =t; tt<T; tt++){
                test_block_itt = arma::find_finite(Design_array.slice(t).row(i));
                G_it = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),tt,
                                         Q_matrix,Design_array,i);
                // Multiply likelihood by density of RT at time tt in t to T
                likelihood_L(cc) = likelihood_L(cc)*dLit(G_it,latency_it.elem(test_block_itt),
                             RT_itempars.rows(test_block_itt),tau_i,phi);
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
          ptransprev(cc) = pTran_HO_joint_g(alpha_pre,alpha_c,lambdas,theta_i,Q_i,Design_array,(t-1),i);
          // Likelihood of RT
          if(G_version!=3){
            if(G_version==1){
              ETA_cc = ETA.col(cc);
              G_it = ETA_cc.elem(block_it);
            }
            if(G_version==2){
              G_it = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),t,
                                       Q_matrix,Design_array,i);
            }



            likelihood_L(cc) = dLit(G_it,latency_it.elem(block_it),RT_itempars.rows(block_it),
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
      post_old += std::log(pTran_HO_joint_g(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_i,Q_i,Design_array,(t-1),i));
      post_new += std::log(pTran_HO_joint_g(alphas.slice(t-1).row(i).t(),alphas.slice(t).row(i).t(),lambdas,theta_new,Q_i,Design_array,(t-1),i));
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
    for (unsigned int t = 0; t<T; t++) {
      block_it = arma::find_finite(Design_array.slice(t).row(i));
      block_it_vec = arma::conv_to<arma::vec>::from(block_it);
      J_it = block_it.n_elem;

      if (G_version == 1) {
        double class_it = arma::dot(alphas.slice(t).row(i).t(), vv);
        arma::vec ETA_it = ETA.col(class_it);
        G_it = ETA_it.elem(block_it);
      }
      if (G_version == 2) {
        G_it = G2vec_efficient_g(ETA, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), t,
                                 Q_matrix, Design_array, i);
      }
      if(G_version==3){
        G_it = arma::ones<arma::vec>(J_it);
        arma::vec y(J_it);
        y.fill((t+1.)/T);
        G_it =G_it % y;
      }


      for (unsigned int j = 0; j<J_it; j++) {
        //add practice*a of (i,j) to num and denom of phi
        block_it_vec.subvec(1,1);
        num += (log(latency(i,block_it_vec(j),t))-RT_itempars(block_it_vec(j), 1)+ phi*G_it(j))*pow(RT_itempars(block_it_vec(j), 0), 2);
        denom += pow(RT_itempars(block_it_vec(j), 0), 2);
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

  // update pi
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
        post_old += std::log(pTran_HO_joint_g(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            lambdas, thetas(i), Q_examinee[i], Design_array, t, i));
        post_new += std::log(pTran_HO_joint_g(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            tmp, thetas(i), Q_examinee[i], Design_array, t, i));
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
  double a_alpha, scl_alpha, mu_gamma, sd_gamma, alpha_sqr, t_star;
  arma::mat Gs(N, J);
  arma::mat ETA_block(N,J);
  arma::vec Class(N);
  arma::vec eta_j(N);
  arma::mat Res_block(N, J);
  arma::mat RT_block(N, J);
  arma::vec alpha;

  for(unsigned int i = 0; i < N; i++){
    tau_i = taus(i);
    for(unsigned int j = 0; j < J; j++){
      // find the time point at which i received this block
      t_star = max(arma::find_finite(Design_array.subcube(i,j,0,i,j,T-1)));
      alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));
      // get response, RT, alphas, and Gs for items in this block
      Res_block.col(j).row(i) = response.slice(t_star).col(j).row(i);
      RT_block.col(j).row(i) = latency.slice(t_star).col(j).row(i);
      arma::vec alpha = alphas.slice(t_star).row(i).t();
      Class(i) = arma::dot(alpha,bijectionvector(K));

      ETA_block(i,j) = ETA(j,Class(i));
      if(G_version == 1){
        Gs(i,j) = ETA_block(i,j);
      }

      if(G_version==3){
        Gs(i,j)= (t_star+1.)/(T);
      }

    }
    if(G_version == 2){
      Gs.row(i) = G2vec_efficient_g(ETA,J_incidence,alphas.subcube(i,0,0,i,(K-1),(T-1)),
             t_star,Q_matrix,Design_array,i).t();
    }
  }

  for(unsigned int j = 0; j < J; j++){
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
    pg = R::pbeta(1.0 - itempars(j,0), ag, bg, 1, 0);
    itempars(j,1) = R::qbeta(ug*pg, ag, bg, 1, 0);
    // update s based on current g
    ps = R::pbeta(1.0 - itempars(j,1), as, bs, 1, 0);
    itempars(j,0) = R::qbeta(us*ps, as, bs, 1, 0);

    // sample the RT model parameters
    arma::vec scl_tmp(N);
    arma::vec mu_tmp(N);
    for(unsigned int i = 0; i < N; i++){
      tau_i = taus(i);
      scl_tmp(i) = pow((log(RT_block(i,j))+tau_i+phi*Gs(i,j)-RT_itempars(j,1)),2);
      mu_tmp(i) = log(RT_block(i,j))+tau_i+phi*Gs(i,j);
    }
    // update alpha_j based on previous gamma_j
    a_alpha = a_alpha0 + N/2;
    // note: the derivation we have corresponds to the rate of gamma, need to take recip for scl
    scl_alpha = 1./(rate_alpha0 + 1./2 * arma::sum(scl_tmp));
    alpha_sqr = R::rgamma(a_alpha,scl_alpha);
    RT_itempars(j,0) = sqrt(alpha_sqr);
    // update gamma_j based on current alpha_j
    mu_gamma = (alpha_sqr*arma::sum(mu_tmp))/(N*alpha_sqr + 1);
    sd_gamma = sqrt(1./(N*alpha_sqr + 1));
    RT_itempars(j,1) = R::rnorm(mu_gamma, sd_gamma);
  }

  // update phi
  double num = 0;
  double denom = 0;
  for(unsigned int i = 0; i<N; i++){
    tau_i = taus(i);
    for(unsigned int t = 0; t<T; t++){

      for(unsigned int j = 0; j<J; j++){
        // find the time point at which i received this item
        t_star = max(arma::find_finite(Design_array.subcube(i,j,0,i,j,T-1)));
        //add practice*a of (i,j,block) to num and denom of phi
        num += pow(RT_itempars(j,0),2) * Gs(i,j)*(log(latency(i,j,t_star))-RT_itempars(j,1)+tau_i);
        denom += pow(RT_itempars(j,0),2)* pow(Gs(i,j),2);
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
Rcpp::List Gibbs_DINA_HO_RT_joint_g(const arma::cube& Response, const arma::cube& Latency,
                                  const arma::mat& Q_matrix, 
                                  const arma::cube& Design_array, int G_version,
                                  const double sig_theta_propose, const arma::vec deltas_propose,
                                  const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Design_array.n_slices;
  unsigned int N = Design_array.n_rows;
  unsigned int K = Q_matrix.n_cols;
  unsigned int J = Q_matrix.n_rows;
  unsigned int nClass = pow(2,K);
  
  Rcpp::List Q_examinee = Q_list_g(Q_matrix, Design_array);
  
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
  arma::cube Alphas_init = simulate_alphas_HO_joint_g(lambdas_init,thetatau_init.col(0),Q_matrix,Design_array,Alphas_0_init);
  
  arma::vec pi_init = rDirichlet(arma::ones<arma::vec>(nClass));
  arma::vec phi_init(1);
  phi_init(0) = R::runif(0,1);
  
  arma::mat itempars_init = .3 * arma::randu<arma::mat>(J,2);
  itempars_init.submat(0,1,J-1,1) =
    itempars_init.submat(0,1,J-1,1) % (1.-itempars_init.submat(0,0,J-1,0));
  arma::mat RT_itempars_init(J,2);
  RT_itempars_init.submat(0,0,J-1,0) = 2.+2.*arma::randu<arma::mat>(J,1);
  RT_itempars_init.submat(0,1,J-1,1) = arma::randn<arma::mat>(J,1)*.5+3.45;
  
  
  
  arma::mat S = arma::eye<arma::mat>(2,2);
  double p = 3.;

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

  
  double tmburn;
  // double deviance;
  double m_accept_theta;
  arma::vec accept_theta_vec, accept_lambdas_vec;
  arma::vec vv = bijectionvector(K*T);
  arma::mat Trajectories_mat(N,(K*T));
  arma::cube ETA, J_incidence;
  
  for(unsigned int tt = 0; tt < chain_length; tt ++){
    Rcpp::List tmp = parm_update_HO_RT_joint_g(Design_array, Alphas_init, pi_init, lambdas_init,
                                             thetas_init, Latency, RT_itempars_init, taus_init,
                                             phi_init, Sig_init, Response, itempars_init, Q_matrix,
                                             Q_examinee, G_version,
                                             sig_theta_propose, S, p, deltas_propose, 1., 1.);
    if(tt>=burn_in){
      tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
      }
      ss.col(tmburn) = itempars_init.col(0);
      gs.col(tmburn) = itempars_init.col(1);
      RT_as.col(tmburn) = RT_itempars_init.col(0);
      RT_gammas.col(tmburn) = RT_itempars_init.col(1);
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
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  Rcpp::List input_data = Rcpp::List::create(Rcpp::Named("Response",Response),
                                             Rcpp::Named("Latency",Latency),
                                             Rcpp::Named("G_version",G_version),
                                             Rcpp::Named("Q_matrix",Q_matrix),
                                             Rcpp::Named("Q_examinee",Q_examinee),
                                             Rcpp::Named("Design_array",Design_array)
  );
  Rcpp::List res = Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
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
                                      Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas),
                                      Rcpp::Named("accept_rate_theta",accept_rate_theta),
                                      Rcpp::Named("accept_rate_lambdas",accept_rate_lambdas),
                                      
                                      Rcpp::Named("Model", "DINA_HO_RT_joint"),
                                      Rcpp::Named("chain_length",chain_length),
                                      Rcpp::Named("burn_in", burn_in),
                                      
                                      Rcpp::Named("input_data",input_data)
  );
  
  res.attr("class") = "hmcdm";
  return res;
}



//[[Rcpp::export]]
void parm_update_rRUM_g(const arma::cube& Design_array,
                        arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, 
                        arma::mat& r_stars, arma::vec& pi_stars, const arma::mat Q_matrix, 
                        const arma::cube& responses, arma::cube& X_ijk, arma::mat& Smats, arma::mat& Gmats,
                        const arma::vec& dirich_prior){
  double N = Design_array.n_rows;
  double J = Design_array.n_cols;
  double T = Design_array.n_slices;
  double K = Q_matrix.n_cols;
  
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj,prodXijk,pi_ijk,aik,u,compare;
  double pi_ik,aik_nmrtr_k,aik_dnmntr_k,c_aik_1,c_aik_0,ptranspost_1,ptranspost_0,ptransprev_1,ptransprev_0;
  arma::vec aik_nmrtr(K);
  arma::vec aik_dnmntr(K);
  // double D_bar = 0;
  arma::mat Classes(N,(T));
  arma::uvec test_block_it;
  
  // update X
  for(unsigned int i=1;i<N;i++){
    for(unsigned int t=0; t<(T); t++){
      test_block_it = arma::find(Design_array.slice(t).row(i) == 1);
      double Jt = arma::sum(Design_array.slice(t).row(i) == 1);
      arma::vec pistar_it = pi_stars.elem(test_block_it);
      arma::mat rstar_it = r_stars.rows(test_block_it);

      arma::vec alpha_i =(alphas.slice(t).row(i)).t();
      arma::vec Yi =(responses.slice(t).row(i)).t();
      
      arma::vec ui = arma::randu<arma::vec>(K);
      aik_nmrtr    = arma::ones<arma::vec>(K);
      aik_dnmntr   = arma::ones<arma::vec>(K);
      
      for(unsigned int j=0;j<Jt;j++){
        // Note that the Xijk cube is indexed from j = 1 to Jk*(T)
        // unsigned int j_star = block*Jt+j;
        unsigned int j_star = test_block_it(j);
        double Yij = Yi(j_star);
        arma::vec Xij = X_ijk.tube(i,j_star); 
        arma::uvec task_ij = find(Q_matrix.row(j_star) == 1);
        
        for(unsigned int k = 0;k<task_ij.n_elem ;k++){
          kj = task_ij(k);
          aik = alpha_i(kj);
          Xij(kj) = 1;
          prodXijk = prod(Xij(task_ij));
          u = R::runif(0.0,1.0);
          pi_ijk = (1.0-prodXijk)*(aik*(1.0-Smats(j_star,kj)) + (1.0-aik)*Gmats(j_star,kj) );
          compare=(pi_ijk>u);
          Xij(kj)=(1.0-Yij)*compare + Yij;
          
          aik_nmrtr(kj) = ( Xij(kj)*(1.0-Smats(j_star,kj)) + (1.0-Xij(kj))*Smats(j_star,kj) )*aik_nmrtr(kj);
          aik_dnmntr(kj) = ( Xij(kj)*Gmats(j_star,kj) + (1.0-Xij(kj))*(1.0-Gmats(j_star,kj)) )*aik_dnmntr(kj);
        }
        X_ijk.tube(i,j_star) = Xij;
      }
      // Rcpp::Rcout<<aik_nmrtr<<std::endl;
      // Rcpp::Rcout<<aik_dnmntr<<std::endl;
      
      //Update alpha_ikt
      for(unsigned int k=0;k<K;k++){
        arma::vec alpha_i_1 = alpha_i;
        alpha_i_1(k) = 1.0;
        c_aik_1 = (arma::as_scalar( alpha_i_1.t()*bijectionvector(K) ));
        arma::vec alpha_i_0 = alpha_i;
        alpha_i_0(k) = 0.0;
        c_aik_0 = (arma::as_scalar( alpha_i_0.t()*bijectionvector(K) ));
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

  for(unsigned int j_star=0;j_star<(J);j_star++){
    // unsigned int test_version_j = floor(j_star/Jt);
    // unsigned int j = j_star % Jt;
    arma::uvec task_ij = find(Q_matrix.row(j_star) == 1);
    arma::mat Xj = X_ijk.tube(0,j_star,N-1,j_star);
    double pistar_temp =1.0;

    arma::mat alpha(N,K);
    for(unsigned int i = 0; i<N; i++){
      // unsigned int t_star = arma::conv_to<unsigned int>::from(arma::find(Test_order.row(Test_versions(i)-1)==(test_version_j+1)));
      // alpha.row(i) = alphas.slice(t_star).row(i);
      unsigned int t_star = max(arma::find_finite(Design_array.subcube(i,j_star,0,i,j_star,T-1)));
      alpha.row(i) = alphas.slice(t_star).row(i);
    }

    for(unsigned int k = 0;k<task_ij.n_elem ;k++){
      kj = task_ij(k);
      arma::vec Xjk = Xj.col(kj);
      arma::vec ak = alpha.col(kj);

      double Sumalphak =  (arma::as_scalar(ak.t() * ak));
      double SumXjk = (arma::as_scalar(Xjk.t() * Xjk));
      double SumXjkalphak = (arma::as_scalar(Xjk.t() * ak));
      double bsk = SumXjkalphak ;
      double ask = Sumalphak - SumXjkalphak ;
      double agk = SumXjk - SumXjkalphak ;
      double bgk = N - SumXjk - Sumalphak + SumXjkalphak ;
      ug = R::runif(0.0,1.0);
      us = R::runif(0.0,1.0);

      //draw g conditoned upon s_t-1
      pg = R::pbeta(1.0-Smats(j_star,kj),agk+1.0,bgk+1.0,1,0);
      gjk = R::qbeta(ug*pg,agk+1.0,bgk+1.0,1,0);
      //draw s conditoned upon g
      ps = R::pbeta(1.0-gjk,ask+1.0,bsk+1.0,1,0);
      sjk = R::qbeta(us*ps,ask+1.0,bsk+1.0,1,0);

      Gmats(j_star,kj) = gjk;
      Smats(j_star,kj) = sjk;

      r_stars(j_star,kj) = gjk/(1.0 - sjk);//compute rstarjk
      pistar_temp = (1.0-sjk)*pistar_temp;//compute pistarj
    }
    pi_stars(j_star) = pistar_temp;
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
Rcpp::List Gibbs_rRUM_indept_g(const arma::cube& Response, const arma::mat& Q_matrix, const arma::mat& R,
                               const arma::cube& Design_array,
                               const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Design_array.n_slices;
  unsigned int N = Response.n_rows;
  unsigned int K = Q_matrix.n_cols;
  unsigned int J = Q_matrix.n_rows;
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
  
  arma::mat r_stars_init = .5 + .2*arma::randu<arma::mat>(J,K);
  arma::vec pi_stars_init = .7 + .2*arma::randu<arma::vec>(J);
  
  arma::mat Smats_init = arma::randu<arma::mat>(J,K);
  arma::mat Gmats_init = arma::randu<arma::mat>(J,K) % (1-Smats_init);
  
  arma::cube X = arma::ones<arma::cube>(N,J,K);
  
  
  // Create objects for storage 
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::cube r_stars(J,K,(chain_length-burn_in));
  arma::mat pi_stars(J,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat taus(K,(chain_length-burn_in));
  
  arma::mat Trajectories_mat(N,(K*T));
  arma::vec vv = bijectionvector(K*T);
  
  
  for(unsigned int tt = 0; tt < chain_length; tt++){
    parm_update_rRUM_g(Design_array,Alphas_init,pi_init,taus_init,R,r_stars_init,pi_stars_init,
                     Q_matrix, Response, X, Smats_init, Gmats_init, 
                     dirich_prior);
    
    if(tt>=burn_in){
      unsigned int tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
      }
      r_stars.slice(tmburn) = r_stars_init;
      pi_stars.col(tmburn) = pi_stars_init;
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      taus.col(tmburn) = taus_init;
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  Rcpp::List input_data = Rcpp::List::create(Rcpp::Named("Response",Response),
                                             Rcpp::Named("Q_matrix",Q_matrix),
                                             Rcpp::Named("Design_array",Design_array),
                                             Rcpp::Named("R",R)
  );
  Rcpp::List res = Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                                      Rcpp::Named("r_stars",r_stars),
                                      Rcpp::Named("pi_stars",pi_stars),
                                      Rcpp::Named("pis",pis),
                                      Rcpp::Named("taus",taus),
                                      
                                      Rcpp::Named("Model", "rRUM_indept"),
                                      Rcpp::Named("chain_length", chain_length),
                                      Rcpp::Named("burn_in", burn_in),
                                      
                                      Rcpp::Named("input_data",input_data)
  );
  
  
  res.attr("class") = "hmcdm";
  return res;
}




//[[Rcpp::export]]
void parm_update_NIDA_indept_g(const arma::cube& Design_array,
                             arma::cube& alphas, arma::vec& pi, arma::vec& taus, const arma::mat& R, const arma::mat Q_matrix, 
                             const arma::cube& responses, arma::cube& X_ijk, arma::mat& Smats, arma::mat& Gmats,
                             const arma::vec& dirich_prior){
  double N = Design_array.n_rows;
  double J = Design_array.n_cols;
  double T = Design_array.n_slices;
  double K = Q_matrix.n_cols;
  
  arma::vec k_index = arma::linspace(0,K-1,K);
  double kj,prodXijk,pi_ijk,aik,u,compare;
  double pi_ik,aik_nmrtr_k,aik_dnmntr_k,c_aik_1,c_aik_0,ptranspost_1,ptranspost_0,ptransprev_1,ptransprev_0;
  arma::vec aik_nmrtr(K);
  arma::vec aik_dnmntr(K);
  // double D_bar = 0;
  arma::mat Classes(N,T);
  arma::uvec test_block_it;
  
  // update X
  for(unsigned int i=1;i<N;i++){
    for(unsigned int t=0; t<(T); t++){
      test_block_it = arma::find(Design_array.slice(t).row(i) == 1);
      double Jt = arma::sum(Design_array.slice(t).row(i) == 1);
      
      arma::vec alpha_i =(alphas.slice(t).row(i)).t();
      arma::vec Yi =(responses.slice(t).row(i)).t();
      
      arma::vec ui = arma::randu<arma::vec>(K);
      aik_nmrtr    = arma::ones<arma::vec>(K);
      aik_dnmntr   = arma::ones<arma::vec>(K);
      
      for(unsigned int j=0;j<Jt;j++){
        unsigned int j_star = test_block_it(j);
        double Yij = Yi(j_star);
        arma::vec Xij = X_ijk.tube(i,j_star); 
        arma::uvec task_ij = find(Q_matrix.row(j_star) == 1);
        // Note that the Xijk cube is indexed from j = 1 to Jk*(T+1)
        
        for(unsigned int k = 0;k<task_ij.n_elem ;k++){
          kj = task_ij(k);
          aik = alpha_i(kj);
          Xij(kj) = 1;
          prodXijk = prod(Xij(task_ij));
          u = R::runif(0.0,1.0);
          pi_ijk = (1.0-prodXijk)*(aik*(1.0-Smats(j_star,kj)) + (1.0-aik)*Gmats(j_star,kj) );
          compare=(pi_ijk>u);
          Xij(kj)=(1.0-Yij)*compare + Yij;
          
          aik_nmrtr(kj) = ( Xij(kj)*(1.0-Smats(j_star,kj)) + (1.0-Xij(kj))*Smats(j_star,kj) )*aik_nmrtr(kj);
          aik_dnmntr(kj) = ( Xij(kj)*Gmats(j_star,kj) + (1.0-Xij(kj))*(1.0-Gmats(j_star,kj)) )*aik_dnmntr(kj);
        }
        X_ijk.tube(i,j_star) = Xij;
      }
      
      //Update alpha_ikt
      for(unsigned int k=0;k<K;k++){
        arma::vec alpha_i_1 = alpha_i;
        alpha_i_1(k) = 1.0;
        c_aik_1 = (arma::as_scalar( alpha_i_1.t()*bijectionvector(K) ));
        arma::vec alpha_i_0 = alpha_i;
        alpha_i_0(k) = 0.0;
        c_aik_0 = (arma::as_scalar( alpha_i_0.t()*bijectionvector(K) ));
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
        
        test_block_it = arma::find(Design_array.slice(t).row(i) == 1);
        double Jt = arma::sum(Design_array.slice(t).row(i) == 1);
        
        for(unsigned int j = 0; j<Jt; j++){
          unsigned int j_star = test_block_it(j);
          Xijk = X_ijk(i,j_star,k); 
          qijk = Q_matrix(j_star,k);
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
    pg = R::pbeta(1.0-Smats(0,k),agk+1.0,bgk+1.0,1,0);
    gk = R::qbeta(ug*pg,agk+1.0,bgk+1.0,1,0);
    //draw s conditoned upon g
    ps = R::pbeta(1.0-gk,ask+1.0,bsk+1.0,1,0);
    sk = R::qbeta(us*ps,ask+1.0,bsk+1.0,1,0);
    
    Gmats.submat(0,k,(J-1),k).fill(gk);
    Smats.submat(0,k,(J-1),k).fill(sk);
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
Rcpp::List Gibbs_NIDA_indept_g(const arma::cube& Response, const arma::mat& Q_matrix, const arma::mat& R,
                             const arma::cube& Design_array,
                             const unsigned int chain_length, const unsigned int burn_in){
  unsigned int T = Design_array.n_slices;
  unsigned int N = Design_array.n_rows;
  unsigned int K = Q_matrix.n_cols;
  unsigned int J = Q_matrix.n_rows;
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
  
  arma::mat r_stars_init = .5 + .2*arma::randu<arma::mat>(J,K);
  arma::vec pi_stars_init = .7 + .2*arma::randu<arma::vec>(J);
  
  arma::mat Smats_init = arma::randu<arma::mat>(J,K);
  arma::mat Gmats_init = arma::randu<arma::mat>(J,K) % (1-Smats_init);
  
  arma::cube X = arma::ones<arma::cube>(N,J,K);
  
  
  // Create objects for storage 
  arma::mat Trajectories(N,(chain_length-burn_in));
  arma::mat Ss(K,(chain_length-burn_in));
  arma::mat Gs(K,(chain_length-burn_in));
  arma::mat pis(nClass, (chain_length-burn_in));
  arma::mat taus(K,(chain_length-burn_in));
  
  arma::mat Trajectories_mat(N,(K*T));
  arma::vec vv = bijectionvector(K*T);
  
  for(unsigned int tt = 0; tt < chain_length; tt++){
    parm_update_NIDA_indept_g(Design_array,Alphas_init,pi_init,taus_init,R,
                            Q_matrix, Response, X, Smats_init, Gmats_init, 
                            dirich_prior);
    
    if(tt>=burn_in){
      unsigned int tmburn = tt-burn_in;
      for(unsigned int t = 0; t < T; t++){
        Trajectories_mat.cols(K*t,(K*(t+1)-1)) = Alphas_init.slice(t);
      }
      Ss.col(tmburn) = Smats_init.row(0).t();
      Gs.col(tmburn) = Gmats_init.row(0).t();
      
      Trajectories.col(tmburn) = Trajectories_mat * vv;
      pis.col(tmburn) = pi_init;
      taus.col(tmburn) = taus_init;
    }
    if(tt%1000==0){
      Rcpp::Rcout<<tt<<std::endl;
    }
  }
  Rcpp::List input_data = Rcpp::List::create(Rcpp::Named("Response",Response),
                                             Rcpp::Named("Q_matrix",Q_matrix),
                                             Rcpp::Named("Design_array",Design_array),
                                             Rcpp::Named("R",R)
  );
  Rcpp::List res = Rcpp::List::create(Rcpp::Named("trajectories",Trajectories),
                                      Rcpp::Named("ss",Ss),
                                      Rcpp::Named("gs",Gs),
                                      Rcpp::Named("pis",pis),
                                      Rcpp::Named("taus",taus),
                                      
                                      Rcpp::Named("Model", "NIDA_indept"),
                                      Rcpp::Named("chain_length", chain_length),
                                      Rcpp::Named("burn_in", burn_in),
                                      
                                      Rcpp::Named("input_data",input_data)
  );
  res.attr("class") = "hmcdm";
  return res;
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
Rcpp::List Gibbs_DINA_FOHM_g(const arma::cube& Response,const arma::mat& Q_matrix,
                             const arma::cube& Design_array,
                             const unsigned int chain_length, const unsigned int burn_in){
  
  unsigned int N = Design_array.n_rows;
  unsigned int J = Design_array.n_cols;
  unsigned int nT = Design_array.n_slices;
  unsigned int K = Q_matrix.n_cols;
  unsigned int C = pow(2,K);
  unsigned int chain_m_burn = chain_length-burn_in;
  unsigned int tmburn;
  
  arma::vec vv = bijectionvector(K);
  arma::mat ETA = ETAmat(K,J,Q_matrix);
  arma::mat TP = TPmat(K);
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
  arma::mat Omega = rOmega(TP);
  arma::vec class0 = arma::randi<arma::vec>(N,arma::distr_param(0,C-1));
  arma::mat CLASS=rAlpha(Omega,N,nT,class0);
  arma::vec ss = arma::randu<arma::vec>(J);
  arma::vec gs = (arma::ones<arma::vec>(J) - ss)%arma::randu<arma::vec>(J);
  arma::vec delta0 = arma::ones<arma::vec>(C);
  arma::vec pis = rDirichlet(delta0);
  
  //Start Markov chain
  for(unsigned int t = 0; t < chain_length; t++){
    parm_update_DINA_FOHM(N,J,K,C,nT,Response,TP,ETA,ss,gs,CLASS,pis,Omega);
    
    if(t>=burn_in){
      tmburn = t-burn_in;
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
  Rcpp::List input_data = Rcpp::List::create(Rcpp::Named("Response",Response),
                                             Rcpp::Named("Q_matrix",Q_matrix),
                                             Rcpp::Named("Design_array",Design_array)
  );
  Rcpp::List res = Rcpp::List::create(Rcpp::Named("ss",SS),
                                      Rcpp::Named("gs",GS),
                                      Rcpp::Named("pis",PIs),
                                      Rcpp::Named("omegas",OMEGAS),
                                      Rcpp::Named("trajectories",Trajectories),
                                      
                                      Rcpp::Named("Model", "DINA_FOHM"),
                                      Rcpp::Named("chain_length", chain_length),
                                      Rcpp::Named("burn_in", burn_in),
                                      
                                      Rcpp::Named("input_data",input_data)
  );
  res.attr("class") = "hmcdm";
  return res;
}



//' @title Gibbs sampler for learning models
//' @description Runs MCMC to estimate parameters of any of the listed learning models. 
//' @param Response An \code{array} of dichotomous item responses. t-th slice is an N-by-J matrix of responses at time t.
//' @param Q_matrix A J-by-K Q-matrix. 
//' @param model A \code{charactor} of the type of model fitted with the MCMC sampler, possible selections are 
//' "DINA_HO": Higher-Order Hidden Markov Diagnostic Classification Model with DINA responses;
//' "DINA_HO_RT_joint": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and joint modeling of latent
//' speed and learning ability; 
//' "DINA_HO_RT_sep": Higher-Order Hidden Markov DCM with DINA responses, log-Normal response times, and separate modeling of latent
//' speed and learning ability; 
//' "rRUM_indept": Simple independent transition probability model with rRUM responses
//' "NIDA_indept": Simple independent transition probability model with NIDA responses
//' "DINA_FOHM": First Order Hidden Markov model with DINA responses
//' @param Design_array An \code{array} of dimension N-by-J-by-L indicating the items assigned (1/0) to each subject at each time point. 
//' Either 'Design_array' or both 'Test_order' & 'Test_versions' need to be provided to run HMCDM. 
//' @param Test_order Optional. A \code{matrix} of the order of item blocks for each test version.
//' @param Test_versions Optional. A \code{vector} of the test version of each learner.
//' @param chain_length An \code{int} of the MCMC chain length.
//' @param burn_in An \code{int} of the MCMC burn-in chain length.
//' @param Latency_array Optional. A \code{array} of the response times. t-th slice is an N-by-J matrix of response times at time t.
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
//' output_FOHM = hmcdm(Y_real_array, Q_matrix, "DINA_FOHM", Design_array, 100, 30)
//' }
//' @export
// [[Rcpp::export]]
Rcpp::List hmcdm(const arma::cube Response, const arma::mat Q_matrix, 
                  const std::string model, 
                  const Rcpp::Nullable<arma::cube> Design_array = R_NilValue,                    const Rcpp::Nullable<arma::mat> Test_order = R_NilValue, 
                  const Rcpp::Nullable<arma::vec> Test_versions = R_NilValue,
                  const unsigned int chain_length = 100, const unsigned int burn_in = 50,
                  const int G_version = NA_INTEGER, 
                  const double theta_propose = 0., 
                  const Rcpp::Nullable<arma::cube> Latency_array = R_NilValue,
                  const Rcpp::Nullable<Rcpp::NumericVector> deltas_propose = R_NilValue,
                  const Rcpp::Nullable<Rcpp::NumericMatrix> R = R_NilValue){
 Rcpp::List output;
 
 arma::cube Design_array_temp;
 if(Design_array.isNotNull()){
   Design_array_temp = Rcpp::as<arma::cube>(Design_array); 
 }
 
 unsigned int T = Response.n_slices;
 unsigned int J = Q_matrix.n_rows;
 unsigned int N = Response.n_rows;
 unsigned int Jt = J/T;
 
 if(Design_array.isNull()){
   if(Test_order.isNull()){Rcpp::stop("Error: argument 'Test_order' is missing");}
   if(Test_versions.isNull()){Rcpp::stop("Error: argument ''Test_versions' is missing");}
   
   arma::mat Test_order_temp = Rcpp::as<arma::mat>(Test_order);
   arma::vec Test_versions_temp = Rcpp::as<arma::vec>(Test_versions);
   
   Design_array_temp = design_array(Test_order_temp, Test_versions_temp, Jt);
 }
 
 arma::cube Latency(N, J, T);
 if(Latency_array.isNotNull()){
   Latency = Rcpp::as<arma::cube>(Latency_array);
 }
 
 if(model == "DINA_HO"){
   output = Gibbs_DINA_HO_g(Response, Q_matrix, Design_array_temp, theta_propose, Rcpp::as<arma::vec>(deltas_propose),
                          chain_length, burn_in);
 }
 if(model == "DINA_HO_RT_joint"){
   output = Gibbs_DINA_HO_RT_joint_g(Response, Latency, Q_matrix, Design_array_temp, G_version,
                                   theta_propose, Rcpp::as<arma::vec>(deltas_propose), chain_length, burn_in);
 }
 if(model == "DINA_HO_RT_sep"){
   output = Gibbs_DINA_HO_RT_sep_g(Response, Latency, Q_matrix, Design_array_temp, G_version,
                                 theta_propose, Rcpp::as<arma::vec>(deltas_propose), chain_length, burn_in);
 }
 if(model == "rRUM_indept"){
   output = Gibbs_rRUM_indept_g(Response, Q_matrix, Rcpp::as<arma::mat>(R),Design_array_temp, chain_length, burn_in);
 }
 if(model == "NIDA_indept"){
   output = Gibbs_NIDA_indept_g(Response, Q_matrix, Rcpp::as<arma::mat>(R), Design_array_temp, chain_length, burn_in);
 }
 if(model == "DINA_FOHM"){
   output = Gibbs_DINA_FOHM_g(Response, Q_matrix, Design_array_temp, chain_length, burn_in);
 }
 
 return(output);
}

