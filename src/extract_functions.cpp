#include <RcppArmadillo.h>
#include "basic_functions.h"
#include "resp_functions.h"
#include "rt_functions.h"
#include "trans_functions.h"
#include "extract_functions.h"

// ------------------ Output Extraction ----------------------------------------------------------
//    Functions for computing point estimates, DIC, and posterior predictive probabilities      
// -----------------------------------------------------------------------------------------------




// [[Rcpp::export]]
Rcpp::List point_estimates_learning(const Rcpp::List output, const std::string model, const unsigned int N,
                                    const unsigned int Jt, const unsigned int K, const unsigned int T,
                                    bool alpha_EAP = true){
  Rcpp::List point_ests;
  // extract common outputs
  arma::mat Traject = Rcpp::as<arma::mat>(output["trajectories"]);
  arma::mat pis = Rcpp::as<arma::mat>(output["pis"]);
  unsigned int n_its = Traject.n_cols;
  
  // compute Alpha_hat
  arma::cube Alphas_est = arma::zeros<arma::cube>(N,K,T);
  arma::mat Alphas_i_mat(n_its,K*T);
  if(alpha_EAP==true){                                   // Compute EAP for alphas
    for(unsigned int i = 0; i<N; i++){
      for(unsigned int tt = 0; tt<n_its; tt++){
        Alphas_i_mat.row(tt) = inv_bijectionvector((K*T),Traject(i,tt)).t();
      }
      for(unsigned int kk = 0; kk<(K*T); kk++){
        if(arma::mean(Alphas_i_mat.col(kk))>.5){
          int k_star = kk % K;
          int t_star = (kk-k_star)/K;
          Alphas_est(i,k_star,t_star) = 1;
        }
      }
    }
  }else{                                                // Find most likely trajectory
    for(unsigned int i= 0; i<N; i++){
      arma::vec traject_sorted = arma::sort(Traject.row(i).t());
      double traject_ML = getMode(traject_sorted,n_its);
      arma::vec alpha_i = inv_bijectionvector((K*T),traject_ML);
      for(unsigned int t = 0; t<T; t++){
        Alphas_est.slice(t).row(i) = alpha_i.subvec(K*t, (K*(t+1)-1)).t();
      }
    }
  }
  arma::vec pis_EAP = arma::mean(pis,1);
  
  //"DINA_HO", "DINA_HO_RT_joint", "DINA_HO_RT_sep", "rRUM_indept","NIDA_indept","DINA_FOHM"
  if(model == "DINA_HO"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("thetas_EAP",thetas_EAP),
                                    Rcpp::Named("lambdas_EAP",lambdas_EAP));
  }
  
  if(model == "DINA_HO_RT_sep"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    
    arma::mat tauvar = Rcpp::as<arma::mat>(output["tauvar"]);
    double tauvar_EAP = arma::mean(tauvar.col(0));
    
    
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("as_EAP",as_EAP),
                                    Rcpp::Named("gammas_EAP",gammas_EAP),
                                    Rcpp::Named("thetas_EAP",thetas_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP),
                                    Rcpp::Named("lambdas_EAP",lambdas_EAP),
                                    Rcpp::Named("phis",phi_EAP),
                                    Rcpp::Named("tauvar_EAP",tauvar_EAP));
    
  }
  
  if(model == "DINA_HO_RT_joint"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    
    arma::cube Sigs = Rcpp::as<arma::cube>(output["Sigs"]);
    arma::mat Sigs_EAP = arma::mean(Sigs, 2);
    
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("as_EAP",as_EAP),
                                    Rcpp::Named("gammas_EAP",gammas_EAP),
                                    Rcpp::Named("thetas_EAP",thetas_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP),
                                    Rcpp::Named("lambdas_EAP",lambdas_EAP),
                                    Rcpp::Named("phis",phi_EAP),
                                    Rcpp::Named("Sigs_EAP",Sigs_EAP));
    
  }
  
  if(model == "rRUM_indept"){
    arma::cube r_stars = Rcpp::as<arma::cube>(output["r_stars"]);
    arma::mat r_stars_EAP = arma::mean(r_stars,2);
    
    arma::mat pi_stars = Rcpp::as<arma::mat>(output["pi_stars"]);
    arma::vec pi_stars_EAP = arma::mean(pi_stars,1);
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("r_stars_EAP",r_stars_EAP),
                                    Rcpp::Named("pi_stars_EAP",pi_stars_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP));
    
  }
  
  if(model == "NIDA_indept"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("taus_EAP",taus_EAP));
    
  }
  
  if(model == "DINA_FOHM"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::cube omegas = Rcpp::as<arma::cube>(output["omegas"]);
    arma::mat omegas_EAP = arma::mean(omegas,2);
    
    point_ests = Rcpp::List::create(Rcpp::Named("Alphas_est",Alphas_est),
                                    Rcpp::Named("pis_EAP",pis_EAP),
                                    Rcpp::Named("ss_EAP",ss_EAP),
                                    Rcpp::Named("gs_EAP",gs_EAP),
                                    Rcpp::Named("omegas_EAP",omegas_EAP));
    
  }
  
  
  return(point_ests);
}


// [[Rcpp::export]]
Rcpp::List Learning_fit(const Rcpp::List output, const std::string model,
                        const arma::cube Y_real_array, const arma::mat Q_matrix,
                        const arma::mat Test_order, const arma::vec Test_versions,
                        const Rcpp::Nullable<Rcpp::List> Q_examinee=R_NilValue,
                        const Rcpp::Nullable<arma::cube> Latency_array = R_NilValue, 
                        const int G_version = NA_INTEGER,
                        const Rcpp::Nullable<Rcpp::NumericMatrix> R = R_NilValue){
  
  unsigned int T = Test_order.n_rows;
  unsigned int J = Q_matrix.n_rows;
  unsigned int Jt = J/T;
  unsigned int K = Q_matrix.n_cols;
  unsigned int N = Test_versions.n_elem;
  arma::cube Latency(N, Jt, T);
  if(Latency_array.isNotNull()){
    arma::cube Latency_temp = Rcpp::as<arma::cube>(Latency_array);
    Latency = Sparse2Dense(Latency_temp, Test_order, Test_versions);
  }
  arma::cube Response = Sparse2Dense(Y_real_array, Test_order, Test_versions);
  arma::cube Qs = Mat2Array(Q_matrix, T);
  arma::mat Traject = Rcpp::as<arma::mat>(output["trajectories"]);
  arma::mat pis = Rcpp::as<arma::mat>(output["pis"]);
  arma::vec pis_EAP = arma::mean(pis,1);
  unsigned int n_its = Traject.n_cols;
  arma::cube Alphas_est = arma::zeros<arma::cube>(N,K,T);
  arma::mat Alphas_i_mat(n_its,K*T);
  for(unsigned int i= 0; i<N; i++){
    arma::vec traject_sorted = arma::sort(Traject.row(i).t());
    double traject_ML = getMode(traject_sorted,n_its);
    arma::vec alpha_i = inv_bijectionvector((K*T),traject_ML);
    for(unsigned int t = 0; t<T; t++){
      Alphas_est.slice(t).row(i) = alpha_i.subvec(K*t, (K*(t+1)-1)).t();
    }
  }
  Rcpp::NumericMatrix DIC(3,5); 
  Rcpp::List posterior_predictives;
  
  arma::vec d_tran(n_its);
  arma::vec d_time(n_its);
  arma::vec d_response(n_its);
  arma::vec d_joint(n_its);
  arma::cube alphas(N,K,T);
  arma::vec G_it(Jt);
  arma::cube total_time_PP(N,T,n_its);
  arma::cube total_score_PP(N,T,n_its);
  arma::mat item_mean_PP(Jt*T, n_its);
  arma::cube item_OR_PP(Jt*T, Jt*T, n_its);
  arma::mat RT_mean_PP(Jt*T,n_its);
  
  arma::vec vv = bijectionvector(K);
  arma::mat ETA = ETAmat(K,J,Q_matrix);
  arma::cube ETAs = Mat2Array(ETA, T);
  arma::mat Y_sim_collapsed(N,Jt*T);
  arma::mat L_sim_collapsed(N,Jt*T);
  
  if(model == "DINA_HO"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // next get itempars and simulated responses
      
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      
      arma::cube Y_sim_sparse = simDINA(alphas,itempars_cube,ETA,Test_order,Test_versions);
      arma::cube Y_sim = Sparse2Dense(Y_sim_sparse, Test_order, Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = Test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas.col(tt), thetas(i,tt),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
          }
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_HO_sep(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        lambdas_EAP, thetas_EAP(i),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
        }
        // The log likelihood from response time model
        int test_block_it = Test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
  }
  
  if(model == "DINA_HO_RT_sep"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    arma::mat tauvar = Rcpp::as<arma::mat>(output["tauvar"]);
    double tauvar_EAP = arma::mean(tauvar.col(0));
    arma::cube J_incidence = J_incidence_cube(Test_order, Qs);
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // put item parameters into a matrix
      // next get itempars and simulated responses
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      arma::mat RT_itempars(Jt*T,2);
      RT_itempars.col(0) = as.col(tt);
      RT_itempars.col(1) = gammas.col(tt);
      arma::cube RT_itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
        RT_itempars_cube.slice(t) = RT_itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      arma::cube Y_sim_sparse = simDINA(alphas,itempars_cube,ETA,Test_order,Test_versions);
      arma::cube Y_sim = Sparse2Dense(Y_sim_sparse, Test_order, Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      arma::cube L_sim_sparse = sim_RT(alphas, RT_itempars_cube,Q_matrix,taus.col(tt),phis(tt,0),
                                ETA, G_version, Test_order, Test_versions);
      arma::cube L_sim = Sparse2Dense(L_sim_sparse, Test_order, Test_versions);
      arma::mat L_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = Test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          L_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = L_sim.slice(t).row(i);
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_HO_sep(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          lambdas.col(tt), thetas(i,tt),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
          }
          if (G_version == 1) {
            G_it = ETAs.slice(test_block_it).col(class_it);
          }
          if (G_version == 2) {
            G_it = G2vec_efficient(ETAs, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                                   Test_order, t);
          }
          if(G_version==3){
            G_it = arma::ones<arma::vec>(Jt);
            arma::vec y(Jt);y.fill((t+1.)/T);
            G_it =G_it % y;
          }
          // The loglikelihood from log-Normal RT model
          time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                                RT_itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                taus(i,tt), phis(tt,0)));
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          total_time_PP(i,t,tt) = arma::sum(L_sim.slice(t).row(i));
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)) + R::dnorm(taus(i,tt),0,std::sqrt(tauvar(tt,0)),true); 
      }
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      RT_mean_PP.col(tt) = arma::mean(L_sim_collapsed,0).t();
      
      
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0) + DIC(0,1) + DIC(0,2) + DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    arma::mat RT_itempars_EAP(Jt*T,2);
    RT_itempars_EAP.col(0) = as_EAP;
    RT_itempars_EAP.col(1) = gammas_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_HO_sep(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        lambdas_EAP, thetas_EAP(i),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
        }
        // The log likelihood from response time model
        int test_block_it = Test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from log-Normal RT model
        time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                              RT_itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                              taus_EAP(i),phi_EAP));
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0))  + R::dnorm(taus_EAP(i),0,std::sqrt(tauvar_EAP),true);
    }
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,1)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("RT_mean_PP",RT_mean_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP),
                                               Rcpp::Named("total_time_PP",total_time_PP));
    
    
  }
  
  if(model == "DINA_HO_RT_joint"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    arma::mat as = Rcpp::as<arma::mat>(output["as"]);
    arma::vec as_EAP = arma::mean(as,1);
    arma::mat gammas = Rcpp::as<arma::mat>(output["gammas"]);
    arma::vec gammas_EAP = arma::mean(gammas,1);
    arma::mat thetas = Rcpp::as<arma::mat>(output["thetas"]);
    arma::vec thetas_EAP = arma::mean(thetas,1);
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    arma::mat lambdas = Rcpp::as<arma::mat>(output["lambdas"]);
    arma::vec lambdas_EAP = arma::mean(lambdas,1);
    arma::mat phis = Rcpp::as<arma::mat>(output["phis"]);
    double phi_EAP = arma::mean(phis.col(0));
    arma::cube Sigs = Rcpp::as<arma::cube>(output["Sigs"]);
    arma::mat Sigs_EAP = arma::mean(Sigs, 2);
    arma::cube J_incidence = J_incidence_cube(Test_order, Qs);
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // put item parameters into a matrix
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      arma::mat RT_itempars(Jt*T,2);
      RT_itempars.col(0) = as.col(tt);
      RT_itempars.col(1) = gammas.col(tt);
      arma::cube RT_itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
        RT_itempars_cube.slice(t) = RT_itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      arma::cube Y_sim_sparse = simDINA(alphas,itempars_cube,ETA,Test_order,Test_versions);
      arma::cube Y_sim = Sparse2Dense(Y_sim_sparse, Test_order, Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      arma::cube L_sim_sparse = sim_RT(alphas, RT_itempars_cube,Q_matrix,taus.col(tt),phis(tt,0),
                                ETA, G_version, Test_order, Test_versions);
      arma::cube L_sim = Sparse2Dense(L_sim_sparse, Test_order, Test_versions);
      arma::mat L_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = Test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          L_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = L_sim.slice(t).row(i);
          
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_HO_joint(alphas.slice(t).row(i).t(),
                                            alphas.slice(t+1).row(i).t(),
                                            lambdas.col(tt), thetas(i,tt),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
          }
          if (G_version == 1) {
            G_it = ETAs.slice(test_block_it).col(class_it);
          }
          if (G_version == 2) {
            G_it = G2vec_efficient(ETAs, J_incidence, alphas.subcube(i, 0, 0, i, (K - 1), (T - 1)), test_version_i,
                                   Test_order, t);
          }
          if(G_version==3){
            G_it = arma::ones<arma::vec>(Jt);
            arma::vec y(Jt);y.fill((t+1.)/T);
            G_it =G_it % y;
          }
          // The loglikelihood from log-Normal RT model
          time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                                RT_itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                taus(i,tt), phis(tt,0)));
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          total_time_PP(i,t,tt) = arma::sum(L_sim.slice(t).row(i));
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        arma::vec thetatau(2);
        thetatau(0) = thetas(i,tt);
        thetatau(1) = taus(i,tt);
        joint += std::log(pis(class_i0,tt)) + std::log(dmvnrm(thetatau,arma::zeros<arma::vec>(2),Sigs.slice(tt),false)); 
      }
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      RT_mean_PP.col(tt) = arma::mean(L_sim_collapsed,0).t();
      
      
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0) + DIC(0,1) + DIC(0,2) + DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    arma::mat RT_itempars_EAP(Jt*T,2);
    RT_itempars_EAP.col(0) = as_EAP;
    RT_itempars_EAP.col(1) = gammas_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_HO_joint(Alphas_est.slice(t).row(i).t(),
                                          Alphas_est.slice(t+1).row(i).t(),
                                          lambdas_EAP, thetas_EAP(i),  Rcpp::as<Rcpp::List>(Q_examinee)[i], Jt, t));
        }
        // The log likelihood from response time model
        int test_block_it = Test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from log-Normal RT model
        time += std::log(dLit(G_it, Latency.slice(t).row(i).t(), 
                              RT_itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                              taus_EAP(i),phi_EAP));
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      arma::vec thetatau(2);
      thetatau(0) = thetas_EAP(i);
      thetatau(1) = taus_EAP(i);
      joint += std::log(pis_EAP(class_i0))  + + std::log(dmvnrm(thetatau,arma::zeros<arma::vec>(2),Sigs_EAP,false));
    }
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,1)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("RT_mean_PP",RT_mean_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP),
                                               Rcpp::Named("total_time_PP",total_time_PP));
    
  }
  
  if(model == "rRUM_indept"){
    arma::cube r_stars = Rcpp::as<arma::cube>(output["r_stars"]);
    arma::mat r_stars_EAP = arma::mean(r_stars,2);
    arma::mat pi_stars = Rcpp::as<arma::mat>(output["pi_stars"]);
    arma::vec pi_stars_EAP = arma::mean(pi_stars,1);
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    for(unsigned int tt = 0; tt < n_its; tt++){
      arma::cube r_stars_cube(Jt,K,T);
      arma::mat pi_stars_mat(Jt,T);
      for(unsigned int t= 0; t<T; t++){
        r_stars_cube.slice(t) = r_stars.slice(tt).rows(Jt*t,(Jt*(t+1)-1));
        pi_stars_mat.col(t) = pi_stars.col(tt).subvec(Jt*t,(Jt*(t+1)-1));
      }
      arma::mat r_stars_mat = Array2Mat(r_stars_cube);
      arma::cube Y_sim_sparse = simrRUM(alphas, r_stars_mat, pi_stars_mat,Q_matrix,Test_order,Test_versions);
      arma::cube Y_sim = Sparse2Dense(Y_sim_sparse, Test_order, Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = Test_order(test_version_i,t)-1;
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_indept(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          taus.col(tt),Rcpp::as<arma::mat>(R)));
          }
          // The loglikelihood from the DINA
          response += std::log(pYit_rRUM(alphas.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                         pi_stars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)).col(tt),
                                         r_stars.slice(tt).rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                         Qs.slice(test_block_it)));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);

    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_indept(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        taus_EAP,Rcpp::as<arma::mat>(R)));
        }
        // The log likelihood from response time model
        int test_block_it = Test_order(test_version_i, t) - 1;
        // The loglikelihood from the DINA
        response += std::log(pYit_rRUM(Alphas_est.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                       pi_stars_EAP.subvec((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                       r_stars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1)),
                                       Qs.slice(test_block_it)));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
    
    
  }
  
  if(model == "NIDA_indept"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::mat taus = Rcpp::as<arma::mat>(output["taus"]);
    arma::vec taus_EAP = arma::mean(taus,1);
    
    
    
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      
      arma::cube Y_sim_sparse = simNIDA(alphas,ss.col(tt),gs.col(tt),Q_matrix,Test_order,Test_versions);
      arma::cube Y_sim = Sparse2Dense(Y_sim_sparse, Test_order, Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = Test_order(test_version_i,t)-1;
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          
          // The transition model part
          if (t < (T - 1)) {
            tran += std::log(pTran_indept(alphas.slice(t).row(i).t(),
                                          alphas.slice(t+1).row(i).t(),
                                          taus.col(tt),Rcpp::as<arma::mat>(R)));
          }
          
          // The loglikelihood from the DINA
          response += std::log(pYit_NIDA(alphas.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                         ss.col(tt),
                                         gs.col(tt),
                                         Qs.slice(test_block_it)));
          
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          tran += std::log(pTran_indept(Alphas_est.slice(t).row(i).t(),
                                        Alphas_est.slice(t+1).row(i).t(),
                                        taus_EAP,Rcpp::as<arma::mat>(R)));
        }
        // The log likelihood from response time model
        int test_block_it = Test_order(test_version_i, t) - 1;
        // The loglikelihood from the DINA
        response += std::log(pYit_NIDA(Alphas_est.slice(t).row(i).t(),Response.slice(t).row(i).t(),
                                       ss_EAP,
                                       gs_EAP,
                                       Qs.slice(test_block_it)));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
    
  }
  
  if(model == "DINA_FOHM"){
    arma::mat ss = Rcpp::as<arma::mat>(output["ss"]);
    arma::vec ss_EAP = arma::mean(ss,1);
    
    arma::mat gs = Rcpp::as<arma::mat>(output["gs"]);
    arma::vec gs_EAP = arma::mean(gs,1);
    
    
    arma::cube omegas = Rcpp::as<arma::cube>(output["omegas"]);
    arma::mat omegas_EAP = arma::mean(omegas,2);
    
    for(unsigned int tt = 0; tt < n_its; tt++){
      double tran=0, response=0, time=0, joint = 0;
      // first get alphas at time tt
      for(unsigned int i = 0; i<N; i++){
        arma::vec alpha_i = inv_bijectionvector((K*T),Traject(i,tt));
        for(unsigned int t = 0; t<T; t++){
          alphas.slice(t).row(i) = alpha_i.subvec((t*K),((t+1)*K-1)).t();
        }
      }
      // put item parameters into a matrix
      arma::mat itempars(Jt*T,2);
      itempars.col(0) = ss.col(tt);
      itempars.col(1) = gs.col(tt);
      arma::cube itempars_cube(Jt,2,T);
      for(unsigned int t= 0; t<T; t++){
        itempars_cube.slice(t) = itempars.rows(Jt*t,(Jt*(t+1)-1));
      }
      arma::cube Y_sim_sparse = simDINA(alphas,itempars_cube,ETA,Test_order,Test_versions);
      arma::cube Y_sim = Sparse2Dense(Y_sim_sparse, Test_order, Test_versions);
      arma::mat Y_sim_collapsed(N,Jt*T);
      
      // next compute deviance part
      for (unsigned int i = 0; i < N; i++) {
        int test_version_i = Test_versions(i) - 1;
        for (unsigned int t = 0; t < T; t++) {
          int test_block_it = Test_order(test_version_i,t)-1;
          double class_it = arma::dot(alphas.slice(t).row(i).t(),vv);
          Y_sim_collapsed.submat(i,(test_block_it*Jt), i, ((test_block_it+1)*Jt-1)) = Y_sim.slice(t).row(i);
          
          // The transition model part
          if (t < (T - 1)) {
            int class_pre, class_post;
            class_pre = arma::dot(alphas.slice(t).row(i),vv);
            class_post = arma::dot(alphas.slice(t+1).row(i),vv);
            tran += std::log(omegas(class_pre,class_post,tt));
          }
          // The loglikelihood from the DINA
          response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                         itempars.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
          total_score_PP(i,t,tt) = arma::sum(Y_sim.slice(t).row(i));
          
          
        }
        double class_i0 = arma::dot(alphas.slice(0).row(i), vv);
        joint += std::log(pis(class_i0,tt)); 
      }
      time = NA_REAL;
      // store dhats for this iteration
      d_tran(tt) = tran;
      d_time(tt) = time;
      d_response(tt) = response;
      d_joint(tt) = joint;
      
      // Posterior predictive
      item_mean_PP.col(tt) = arma::mean(Y_sim_collapsed,0).t();
      item_OR_PP.slice(tt) = OddsRatio(N,Jt*T,Y_sim_collapsed);
      
    }
    DIC(0,0) = -2. * arma::mean(d_tran);
    DIC(0,1) = -2. * arma::mean(d_time);
    DIC(0,2) = -2. * arma::mean(d_response);
    DIC(0,3) = -2. * arma::mean(d_joint);
    DIC(0,4) = DIC(0,0)+DIC(0,2)+DIC(0,3);
    
    // get Dhat
    double tran=0, response=0, time=0, joint = 0;
    arma::mat itempars_EAP(Jt*T,2);
    itempars_EAP.col(0) = ss_EAP;
    itempars_EAP.col(1) = gs_EAP;
    for (unsigned int i = 0; i < N; i++) {
      int test_version_i = Test_versions(i) - 1;
      for (unsigned int t = 0; t < T; t++) {
        
        // The transition model part
        if (t < (T - 1)) {
          int class_pre, class_post;
          class_pre = arma::dot(Alphas_est.slice(t).row(i),vv);
          class_post = arma::dot(Alphas_est.slice(t+1).row(i),vv);
          tran += std::log(omegas_EAP(class_pre,class_post));
        }
        // The log likelihood from response time model
        int test_block_it = Test_order(test_version_i, t) - 1;
        double class_it = arma::dot(Alphas_est.slice(t).row(i), vv);
        // The loglikelihood from the DINA
        response += std::log(pYit_DINA(ETAs.slice(test_block_it).col(class_it), Response.slice(t).row(i).t(), 
                                       itempars_EAP.rows((test_block_it*Jt),((test_block_it+1)*Jt-1))));
        
      }
      double class_i0 = arma::dot(Alphas_est.slice(0).row(i), vv);
      joint += std::log(pis_EAP(class_i0)) ;
    }
    time = NA_REAL;
    DIC(1,0) = -2. * tran;
    DIC(1,1) = -2. * time;
    DIC(1,2) = -2. * response;
    DIC(1,3) = -2. * joint;
    DIC(1,4) = DIC(1,0)+DIC(1,2)+DIC(1,3);
    
    posterior_predictives = Rcpp::List::create(Rcpp::Named("item_mean_PP", item_mean_PP),
                                               Rcpp::Named("item_OR_PP",item_OR_PP),
                                               Rcpp::Named("total_score_PP",total_score_PP));
    
  }
  // get last row --- DIC 
  DIC.row(2) = 2. * DIC.row(0) - DIC.row(1);
  
  Rcpp::List dimnms = Rcpp::List::create(Rcpp::CharacterVector::create("D_bar","D(theta_bar)","DIC"),
                                                                       Rcpp::CharacterVector::create("Transition", "Response_Time","Response","Joint","Total"));
  
  DIC.attr("dimnames") = dimnms;
  
  return Rcpp::List::create(Rcpp::Named("DIC",DIC),
                            Rcpp::Named("PPs",posterior_predictives));
  
  
}
