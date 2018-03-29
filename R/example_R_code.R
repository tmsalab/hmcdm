#' @examples 
#' \dontrun{
#' 
#' #######################################################
#' #   Testing CDM Learning functions                    #
#' #######################################################
#' # Load the spatial rotation data
#' data("Spatial_Rotation")
#' 
#' # Create the ideal response cube, where each slice is the ideal response matrix for an item block
#' ETAs <- array(NA,dim = c(Jt,2^K,T)) 
#' for(t in 1:T){
#'   ETAs[,,t] <- ETAmat(K,Jt,Q_list[[t]])
#' }
#' 
#' # Get the Q matrix for the entire test
#' Q_test <- matrix(NA,J,K)
#' for(t in 1:T){
#'   Q_test[(Jt*(t-1)+1):(Jt*t),] <- Q_list[[t]]
#' }
#' 
#' # -------------- 1. HMDCM -----------------------------------------------------------------------------
#' 
#' # (1) Simulate responses based on the HMDCM model
#' 
#' class_0 <- sample(1:2^K, N, replace = T)
#' Alphas_0 <- matrix(0,N,K)
#' 
#' thetas_true = rnorm(N)
#' 
#' for(i in 1:N){
#'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
#' }
#' 
#' lambdas_true = c(-1, 1.8, .277, .055)
#' 
#' Alphas <- simulate_alphas_HO_sep(lambdas_true,thetas_true,Alphas_0,Q_examinee,T,Jt)
#' table(rowSums(Alphas[,,5]) - rowSums(Alphas[,,1])) # used to see how much transition has taken place
#' 
#' itempars_true <- array(runif(Jt*2*T,.1,.2), dim = c(Jt,2,T))
#' 
#' ETAs <- array(NA,dim = c(Jt,2^K,T)) 
#' for(t in 1:T){
#'   ETAs[,,t] <- ETAmat(K,Jt,Q_list[[t]])
#' }
#' 
#' Y_sim <- simDINA(Alphas,itempars_true,ETAs,test_order,Test_versions)
#' Y_sim_list <- list()
#' for(t in 1:T){
#'   Y_sim_list[[t]] = Y_sim[,,t]
#' }
#' 
#' 
#' (2) Run the MCMC to sample parameters from the posterior distribution
#' 
#' output_HMDCM = MCMC_learning(Y_sim_list,Q_list,"DINA_HO",test_order,Test_versions,10000,5000,
#'                               Q_examinee = Q_examinee,theta_propose = 2,deltas_propose = c(.45,.35,.25,.06))
#' 
#' 
#' 
#' (3) Compute the point estimates of parameters based on the MCMC samples
#' 
#' point_estimates <- point_estimates_learning(output_HMDCM,"DINA_HO",N,Jt,K,T,alpha_EAP = F)
#' 
#' 
#' (4) Evaluate the accuracy of estimated parameters
#' 
#' # Attribute-wise agreement rate between true and estimated alphas
#' AAR_vec <- numeric(T)
#' for(t in 1:T){
#'   AAR_vec[t] <- mean(Alphas[,,t]==point_estimates$Alphas_est[,,t])
#' }
#' 
#' # Pattern-wise agreement rate between true and estimated alphas
#' PAR_vec <- numeric(T)
#' for(t in 1:T){
#'   PAR_vec[t] <- mean(rowSums((Alphas[,,t]-point_estimates$Alphas_est[,,t])^2)==0)
#' }
#' 
#' (5) Evaluate the fit of the model to the observed data (here, simulated data)
#' 
#' # Get DIC and posterior predictive item and subject moments
#' HMDCM_fit <- Learning_fit(output_HMDCM,"DINA_HO",Y_sim_list,Q_list,test_order,Test_versions,Q_examinee)
#' 
#' # To get the posterior predictive probabilities:
#' # <1> For total scores of each person at each time point: 
#' PPP_total_scores <- matrix(NA,N,T)
#' for(i in 1:N){
#'   for(t in 1:T){
#'     f <- ecdf(HMDCM_fit$PPs$total_score_PP[i,t,])
#'     tot_it = sum(Y_sim_list[[t]][i,])
#'     PPP_total_scores[i,t] = f(tot_it)
#'   }
#' }
#' 
#' # <2> For item response means (% correct) and pairwise odds ratios
#' PPP_item_means <- numeric(J)
#' PPP_item_ORs <- matrix(NA,J,J)
#' # Get collapsed response matrix (N*J, each column corresponds to an item)
#' Y_sim_collapsed <- matrix(NA,N,J)
#' for(i in 1:N){
#'   test_i <- Test_versions[i]
#'   for(t in 1:T){
#'     t_i = test_order[test_i,t]
#'     Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_list[[t]][i,]
#'   }
#' }
#' Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
#' for(j in 1:J){
#'   f1 <- ecdf(HMDCM_fit$PPs$item_mean_PP[j,])
#'   mean_obs <- mean(Y_sim_collapsed[,j])
#'   PPP_item_means[j] = f1(mean_obs)
#' }
#' for(j in 1:(J-1)){
#'   for(jp in (j+1):J){
#'     f2 <- ecdf(HMDCM_fit$PPs$item_OR_PP[j,jp,])
#'     PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
#'   }
#' }
#' 
#' 
#' 
#' 
#' 
#' 
#' -------------- 2. HO_RT_sep --------------------------------------------------------------------------
#' 
#' # (1) Simulate responses and response times based on the HMDCM model with response times (no covariance between speed and learning ability)
#' class_0 <- sample(1:2^K, N, replace = T)
#' Alphas_0 <- matrix(0,N,K)
#' 
#' for(i in 1:N){
#'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
#' }
#' 
#' 
#' thetas_true = rnorm(N,0,1)
#' 
#' tausd_true=0.5
#' 
#' taus_true = rnorm(N,0,tausd_true)
#' 
#' G_version = 3
#' 
#' phi_true = 0.8
#' 
#' lambdas_true <- c(-2, 1.6, .4, .055)       # empirical from Wang 2017
#' 
#' Alphas <- simulate_alphas_HO_sep(lambdas_true,thetas_true,Alphas_0,Q_examinee,T,Jt)
#' 
#' table(rowSums(Alphas[,,5]) - rowSums(Alphas[,,1])) # used to see how much transition has taken place
#' 
#' itempars_true <- array(runif(J*2,0.1,0.3), dim = c(Jt,2,T))
#' 
#' RT_itempars_true <- array(NA, dim = c(Jt,2,T))
#' 
#' RT_itempars_true[,2,] <- rnorm(Jt*T,3.45,.5)
#' 
#' RT_itempars_true[,1,] <- runif(Jt*T,1.5,2)
#' 
#' Y_sim <- simDINA(Alphas,itempars_true,ETAs,test_order,Test_versions)
#' Y_sim_list <- list()
#' for(t in 1:T){
#'   Y_sim_list[[t]] = Y_sim[,,t]
#' }
#' 
#' L_sim <- sim_RT(Alphas,RT_itempars_true,Qs,taus_true,phi_true,ETAs,G_version,test_order,Test_versions)
#' L_sim_list <- list()
#' for(t in 1:T){
#'   L_sim_list[[t]] = L_sim[,,t]
#' }
#' 
#' 
#' # (2) Run the MCMC to sample parameters from the posterior distribution
#' 
#' output_HMDCM_RT_sep = MCMC_learning(Y_sim_list,Q_list,"DINA_HO_RT_sep",test_order,Test_versions,5000,2500,
#'                                      Q_examinee = Q_examinee,Latency_list = L_sim_list, G_version = G_version,
#'                                      theta_propose = 2,deltas_propose = c(.45,.35,.25,.06))
#' 
#' # (3) Obtain point estimates based on MCMC samples
#' point_estimates = point_estimates_learning(output_HMDCM_RT_sep,"DINA_HO_RT_sep",N,Jt,K,T,alpha_EAP = T)
#' 
#' # (4) Check for parameter estimation accuracy
#' 
#' cor_thetas <- cor(thetas_true,point_estimates$thetas_EAP)
#' cor_taus <- cor(taus_true,point_estimates$taus_EAP)
#' 
#' cor_ss <- cor(as.vector(itempars_true[,1,]),point_estimates$ss_EAP)
#' cor_gs <- cor(as.vector(itempars_true[,2,]),point_estimates$gs_EAP)
#' 
#' AAR_vec <- numeric(T)
#' for(t in 1:T){
#'   AAR_vec[t] <- mean(Alphas[,,t]==point_estimates$Alphas_est[,,t])
#' }
#' 
#' PAR_vec <- numeric(T)
#' for(t in 1:T){
#'   PAR_vec[t] <- mean(rowSums((Alphas[,,t]-point_estimates$Alphas_est[,,t])^2)==0)
#' }
#' 
#' 
#' # (5) Evaluate the fit of the model to the observed response and response times data (here, Y_sim and R_sim)
#' 
#' # Get DIC and the posteiror predictive person and item moments
#' HO_RT_sep_fit <- Learning_fit(output_HMDCM_RT_sep,"DINA_HO_RT_sep",Y_sim_list,Q_list,
#'                               test_order,Test_versions,Q_examinee,L_sim_list,G_version)
#' 
#' # To get the posterior predictive probabilities:
#' # <1> For total scores and total response times of each person at each time point: 
#' PPP_total_scores <- PPP_total_RTs <- matrix(NA,N,T)
#' for(i in 1:N){
#'   for(t in 1:T){
#'     f1 <- ecdf(HO_RT_sep_fit$PPs$total_score_PP[i,t,])
#'     tot_score_it = sum(Y_sim_list[[t]][i,])
#'     PPP_total_scores[i,t] = f1(tot_score_it)
#'     
#'     f2 <- ecdf(HO_RT_sep_fit$PPs$total_time_PP[i,t,])
#'     tot_time_it = sum(L_sim_list[[t]][i,])
#'     PPP_total_RTs[i,t] = f2(tot_time_it)
#'   }
#' }
#' 
#' 
#' # <2> For item response and response time means (% correct) and item response pairwise odds ratios
#' PPP_item_means <- PPP_item_mean_RTs <- numeric(J)
#' PPP_item_ORs <- matrix(NA,J,J)
#' # Get collapsed response/RT matrix (N*J, each column corresponds to an item)
#' Y_sim_collapsed <- L_sim_collapsed <- matrix(NA,N,J)
#' for(i in 1:N){
#'   test_i <- Test_versions[i]
#'   for(t in 1:T){
#'     t_i = test_order[test_i,t]
#'     Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_list[[t]][i,]
#'     L_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- L_sim_list[[t]][i,]
#'   }
#' }
#' Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
#' for(j in 1:J){
#'   f1 <- ecdf(HO_RT_sep_fit$PPs$item_mean_PP[j,])
#'   mean_obs <- mean(Y_sim_collapsed[,j])
#'   PPP_item_means[j] = f1(mean_obs)
#'   
#'   f3 <- ecdf(HO_RT_sep_fit$PPs$RT_mean_PP[j,])
#'   mean_RT_obs <- mean(L_sim_collapsed[,j])
#'   PPP_item_mean_RTs[j] = f3(mean_RT_obs)
#' }
#' for(j in 1:(J-1)){
#'   for(jp in (j+1):J){
#'     f2 <- ecdf(HO_RT_sep_fit$PPs$item_OR_PP[j,jp,])
#'     PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
#'   }
#' }
#' 
#' # -------------- 3. HO_RT_joint ----------------------------------------------------------
#' 
#' class_0 <- sample(1:2^K, N, replace = T)
#' Alphas_0 <- matrix(0,N,K)
#' 
#' mu_thetatau = c(0,0)
#' Sig_thetatau = rbind(c(1.8^2,.4*.5*1.8),c(.4*.5*1.8,.25))
#' Z = matrix(rnorm(N*2),N,2)
#' thetatau_true = Z%*%chol(Sig_thetatau)
#' thetas_true = thetatau_true[,1]
#' taus_true = thetatau_true[,2]
#' 
#' G_version = 3
#' 
#' phi_true = 0.8
#' 
#' for(i in 1:N){
#'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
#' }
#' 
#' 
#' lambdas_true <- c(-2, .4, .055)       # empirical from Wang 2017
#' 
#' Alphas <- simulate_alphas_HO_joint(lambdas_true,thetas_true,Alphas_0,Q_examinee,T,Jt)
#' 
#' table(rowSums(Alphas[,,5]) - rowSums(Alphas[,,1])) # used to see how much transition has taken place
#' 
#' 
#' itempars_true <- array(runif(J*2,0.1,0.3), dim = c(Jt,2,T))
#' 
#' RT_itempars_true <- array(NA, dim = c(Jt,2,T))
#' 
#' RT_itempars_true[,2,] <- rnorm(Jt*T,3.45,.5)
#' 
#' RT_itempars_true[,1,] <- runif(Jt*T,1.5,2)
#' 
#' Y_sim <- simDINA(Alphas,itempars_true,ETAs,test_order,Test_versions)
#' Y_sim_list <- list()
#' for(t in 1:T){
#'   Y_sim_list[[t]] = Y_sim[,,t]
#' }
#' 
#' L_sim <- sim_RT(Alphas,RT_itempars_true,Qs,taus_true,phi_true,ETAs,G_version,test_order,Test_versions)
#' L_sim_list <- list()
#' for(t in 1:T){
#'   L_sim_list[[t]] = L_sim[,,t]
#' }
#' 
#' 
#' output_HMDCM_RT_joint = MCMC_learning(Y_sim_list,Q_list,"DINA_HO_RT_joint",test_order,Test_versions,10000,5000,
#'                                  Q_examinee = Q_examinee,Latency_list = L_sim_list, G_version = G_version,
#'                                  theta_propose = 2,deltas_propose = c(.45,.25,.06))
#' 
#' 
#' point_estimates = point_estimates_learning(output_HMDCM_RT_joint,"DINA_HO_RT_joint",N,Jt,K,T,alpha_EAP = T)
#' 
#' 
#' cor_thetas <- cor(thetas_true,point_estimates$thetas_EAP)
#' cor_taus <- cor(taus_true,point_estimates$taus_EAP)
#' 
#' cor_ss <- cor(as.vector(itempars_true[,1,]),point_estimates$ss_EAP)
#' 
#' cor_gs <- cor(as.vector(itempars_true[,2,]),point_estimates$gs_EAP)
#' 
#' AAR_vec <- numeric(T)
#' for(t in 1:T){
#'   AAR_vec[t] <- mean(Alphas[,,t]==point_estimates$Alphas_est[,,t])
#' }
#' 
#' PAR_vec <- numeric(T)
#' for(t in 1:T){
#'   PAR_vec[t] <- mean(rowSums((Alphas[,,t]-point_estimates$Alphas_est[,,t])^2)==0)
#' }
#' 
#' HO_RT_joint_fit <- Learning_fit(output_HMDCM_RT_joint,"DINA_HO_RT_joint",Y_sim_list,Q_list,test_order,Test_versions,Q_examinee,L_sim_list,G_version)
#' 
#' # For how to calculate the posterior predictive probabilities, see the example of DINA_HO_RT_sep
#' 
#' 
#' # -------------- 4. rRUM Independent -----------------------------------------
#' 
#' nClass = 2^K
#' 
#' # Reachability matrix
#' R <- matrix(0,K,K)
#' 
#' tau <- numeric(K)
#' for(k in 1:K){
#'   tau[k] <- runif(1,.2,.6)
#' }
#' # Initial alphas
#' p_mastery <- c(.5,.5,.4,.4)
#' Alphas_0 <- matrix(0,N,K)
#' for(i in 1:N){
#'   for(k in 1:K){
#'     prereqs <- which(R[k,]==1)
#'     if(length(prereqs)==0){
#'       Alphas_0[i,k] <- rbinom(1,1,p_mastery[k])
#'     }
#'     if(length(prereqs)>0){
#'       Alphas_0[i,k] <- prod(Alphas_0[i,prereqs])*rbinom(1,1,p_mastery)
#'     }
#'   }
#' }
#' 
#' # Subsequent Alphas
#' Alphas <- simulate_alphas_indept(tau,Alphas_0,T,R)
#' table(rowSums(Alphas[,,5]) - rowSums(Alphas[,,1])) # used to see how much transition has taken place
#' 
#' Smats <- array(runif(Jt*K*(T),.1,.3),c(Jt,K,(T)))
#' Gmats <- array(runif(Jt*K*(T),.1,.3),c(Jt,K,(T)))
#' 
#' 
#' # Simulate rrum parameters
#' r_stars <- array(NA,c(Jt,K,T))
#' pi_stars <- matrix(NA,Jt,(T))
#' pi_star <- r_star <- list()
#' 
#' for(t in 1:T){
#'   pi_star[[t]] <- apply(((1-Smats[,,t])^Qs[,,t]),1,prod)
#'   r_star[[t]] <- Gmats[,,t]/(1-Smats[,,t])
#' }
#' 
#' 
#' for(t in 1:T){
#'   pi_stars[,t] <- pi_star[[t]]
#'   r_stars[,,t] <- r_star[[t]]
#' }
#' 
#' Test_versions_sim <- sample(1:5,N,replace = T)
#' 
#' Y_sim = simrRUM(Alphas,r_stars,pi_stars,Qs,test_order,Test_versions_sim)
#' 
#' Y_sim_list <- list()
#' for(t in 1:T){
#'   Y_sim_list[[t]] = Y_sim[,,t]
#' }
#' 
#' output_rRUM_indept = MCMC_learning(Y_sim_list,Q_list,"rRUM_indept",test_order,Test_versions_sim,30000,15000,
#'                                     R = R)
#' 
#' point_estimates = point_estimates_learning(output_rRUM_indept,"rRUM_indept",N,Jt,K,T,alpha_EAP = T)
#' 
#' 
#' cor_pistars <- cor(as.vector(pi_stars),point_estimates$pi_stars_EAP)
#' 
#' r_stars_mat <- matrix(NA,Jt*T, K)
#' for(t in 1:T){
#'   r_stars_mat[(Jt*(t-1)+1):(Jt*t),] <- r_stars[,,t]
#' }
#' 
#' cor_rstars <- cor(as.vector(r_stars_mat*Q_test),as.vector(point_estimates$r_stars_EAP*Q_test))
#' 
#' AAR_vec <- numeric(T)
#' for(t in 1:T){
#'   AAR_vec[t] <- mean(Alphas[,,t]==point_estimates$Alphas_est[,,t])
#' }
#' 
#' PAR_vec <- numeric(T)
#' for(t in 1:T){
#'   PAR_vec[t] <- mean(rowSums((Alphas[,,t]-point_estimates$Alphas_est[,,t])^2)==0)
#' }
#' 
#' rRUM_indept_fit <- Learning_fit(output_rRUM_indept,"rRUM_indept",Y_sim_list,Q_list,test_order,Test_versions_sim,R = matrix(0,K,K))
#' 
#' # For how to calculate the posterior predictive probabilities, see the example of HMDCM (DINA_HO)
#' 
#' # -------------- 5. NIDA Independent ------------------------------------------
#' 
#' Test_versions_sim <- sample(1:5,N,replace = T)
#' 
#' nClass = 2^K
#' R <- matrix(0,K,K)
#' Qs <- r_stars <- array(NA,c(Jt,K,T))
#' pi_stars <- matrix(NA,Jt,(T))
#' tau <- numeric(K)
#' 
#' for(k in 1:K){
#'   tau[k] <- runif(1,.2,.6)
#' }
#' # Initial Alphas
#' p_mastery <- c(.5,.5,.4,.4)
#' Alphas_0 <- matrix(0,N,K)
#' for(i in 1:N){
#'   for(k in 1:K){
#'     prereqs <- which(R[k,]==1)
#'     if(length(prereqs)==0){
#'       Alphas_0[i,k] <- rbinom(1,1,p_mastery[k])
#'     }
#'     if(length(prereqs)>0){
#'       Alphas_0[i,k] <- prod(Alphas_0[i,prereqs])*rbinom(1,1,p_mastery)
#'     }
#'   }
#' }
#' # Subsequent Alphas
#' Alphas <- simulate_alphas_indept(tau,Alphas_0,T,R)
#' table(rowSums(Alphas[,,5]) - rowSums(Alphas[,,1])) # used to see how much transition has taken place
#' for(t in 1:(T)){
#'   Qs[,,t] <- Q_list[[t]]
#' }
#' Gmats <- array(NA,c(Jt,K,T))
#' Smats <- array(NA,c(Jt,K,T))
#' for(k in 1:K){
#'   Smats[,k,] <- runif(1,.1,.3)
#'   Gmats[,k,] <- runif(1,.1,.3)
#' }
#' 
#' # Simulate rrum parameters
#' pi_star <- r_star <- list()
#' 
#' for(t in 1:(T)){
#'   pi_star[[t]] <- apply(((1-Smats[,,t])^Qs[,,t]),1,prod)
#'   r_star[[t]] <- Gmats[,,t]/(1-Smats[,,t])
#' }
#' 
#' 
#' for(t in 1:(T)){
#'   pi_stars[,t] <- pi_star[[t]]
#'   r_stars[,,t] <- r_star[[t]]
#' }
#' 
#' Y_sim = simNIDA(Alphas,Smats[1,,1],Gmats[1,,1],Qs,test_order,Test_versions_sim)
#' 
#' Y_sim_list <- list()
#' for(t in 1:T){
#'   Y_sim_list[[t]] = Y_sim[,,t]
#' }
#' 
#' output_NIDA_indept = MCMC_learning(Y_sim_list,Q_list,"NIDA_indept",test_order,Test_versions_sim,30000,15000,
#'                                     R = R)
#' 
#' point_estimates = point_estimates_learning(output_NIDA_indept,"NIDA_indept",N,Jt,K,T,alpha_EAP = T)
#' 
#' 
#' 
#' AAR_vec <- numeric(T)
#' for(t in 1:T){
#'   AAR_vec[t] <- mean(Alphas[,,t]==point_estimates$Alphas_est[,,t])
#' }
#' 
#' PAR_vec <- numeric(T)
#' for(t in 1:T){
#'   PAR_vec[t] <- mean(rowSums((Alphas[,,t]-point_estimates$Alphas_est[,,t])^2)==0)
#' }
#' 
#' NIDA_indept_fit <- Learning_fit(output_NIDA_indept,"NIDA_indept",Y_sim_list,Q_list,test_order,Test_versions_sim,R = matrix(0,K,K))
#' 
#' # For how to calculate the posterior predictive probabilities, see the example of HMDCM (DINA_HO)
#' 
#' # -------------- 6. FOHM ---------------------------------------
#' TP <- TPmat(K)
#' 
#' Omega_true <- rOmega(TP)
#' 
#' class_0 <- sample(1:2^K, N, replace = T)
#' Alphas_0 <- matrix(0,N,K)
#' 
#' for(i in 1:N){
#'   Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
#' }
#' 
#' Alphas <- simulate_alphas_FOHM(Omega_true, Alphas_0,T)
#' 
#' itempars_true <- array(runif(Jt*2*T,.1,.2), dim = c(Jt,2,T))
#' 
#' Y_sim <- simDINA(Alphas,itempars_true,ETAs,test_order,Test_versions)
#' Y_sim_list <- list()
#' for(t in 1:T){
#'   Y_sim_list[[t]] = Y_sim[,,t]
#' }
#' 
#' output_FOHM = MCMC_learning(Y_sim_list,Q_list,"DINA_FOHM",test_order,Test_versions,10000,5000)
#' 
#' point_estimates = point_estimates_learning(output_FOHM,"DINA_FOHM",N,Jt,K,T,alpha_EAP = T)
#' 
#' AAR_vec <- numeric(T)
#' for(t in 1:T){
#'   AAR_vec[t] <- mean(Alphas[,,t]==point_estimates$Alphas_est[,,t])
#' }
#' 
#' PAR_vec <- numeric(T)
#' for(t in 1:T){
#'   PAR_vec[t] <- mean(rowSums((Alphas[,,t]-point_estimates$Alphas_est[,,t])^2)==0)
#' }
#' 
#' FOHM_fit <- Learning_fit(output_FOHM,"DINA_FOHM",Y_sim_list,Q_list,test_order,Test_versions)
#' 
#' # For how to calculate the posterior predictive probabilities, see the example of HMDCM (DINA_HO)
#' }