

#' @export
print.hmcdm <- function(x, ...){
  N <- dim(x$input_data$Response)[1]
  Jt <- dim(x$input_data$Qs)[1]
  K <- dim(x$input_data$Qs)[2]
  T <- dim(x$input_data$Qs)[3]
  J <- Jt*T
  cat("\nModel:",formatC(x$Model),"\n")
  
  cat("\nSample Size:", N)
  cat("\nNumber of Items:", J)
  cat("\nNumber of Time Points:", T,"\n")
  
  cat("\nChain Length:",x$chain_length)
  cat(", burn-in:",x$burn_in,"\n")
}


#' @name summary.hmcdm
#' @title Summarizing Hidden Markov Cognitive Diagnosis Model Fits
#' @description `summary` method for class "`hmcdm`" or "`summary.hmcdm`".
#' @param object a fitted model object of class "`hmcdm`".
#' @param ... further arguments passed to or from other methods.
#' @return The function `summary.hmcdm` computes and returns a \code{list} of point estimates of model parameters and model fit measures including DIC and PPP-values.
#' @seealso [hmcdm()]
#' @examples
#' \donttest{
#' output_FOHM = hmcdm(Y_real_array,Q_matrix,"DINA_FOHM",Test_order,Test_versions,10000,5000)
#' summary(output_FOHM)
#' }
#' @export
summary.hmcdm <- function(object, ...){
  N <- nrow(object$input_data$Response)[1]
  Jt <- dim(object$input_data$Qs)[1]
  K <- dim(object$input_data$Qs)[2]
  T <- dim(object$input_data$Qs)[3]
  J <- Jt*T
  # DINA_HO
  if(object$Model == "DINA_HO"){
    Y_sim <- Dense2Sparse(object$input_data$Response, object$input_data$Test_order, object$input_data$Test_versions)
    Q_matrix <- object$input_data$Qs[,,1]
    if(T > 1){
      for(i in 2:T){
        Q_matrix <- rbind(Q_matrix, object$input_data$Qs[,,i])
      }
    }
    point_estimates <- point_estimates_learning(object,"DINA_HO",N,Jt,K,T,alpha_EAP = F)
    HMDCM_fit <- Learning_fit(object, "DINA_HO", Y_sim,Q_matrix, 
                              object$input_data$Test_order, object$input_data$Test_versions, object$input_data$Q_examinee)
    PPP_total_scores <- matrix(NA,N,T)
    Y_sim_array <- Sparse2Dense(Y_sim, object$input_data$Test_order, object$input_data$Test_versions)
    for(i in 1:N){
      for(t in 1:T){
        f <- stats::ecdf(HMDCM_fit$PPs$total_score_PP[i,t,])
        tot_it = sum(Y_sim_array[i,,t])
        PPP_total_scores[i,t] = f(tot_it)
      }
    }
    PPP_item_means <- numeric(J)
    PPP_item_ORs <- matrix(NA,J,J)
    Y_sim_collapsed <- matrix(NA,N,J)
    for(i in 1:N){
      test_i <- object$input_data$Test_versions[i]
      for(t in 1:T){
        t_i = object$input_data$Test_order[test_i,t]
        Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_array[i,,t]
      }
    }
    Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
    for(j in 1:J){
      f1 <- stats::ecdf(HMDCM_fit$PPs$item_mean_PP[j,])
      mean_obs <- mean(Y_sim_collapsed[,j])
      PPP_item_means[j] = f1(mean_obs)
    }
    for(j in 1:(J-1)){
      for(jp in (j+1):J){
        f2 <- stats::ecdf(HMDCM_fit$PPs$item_OR_PP[j,jp,])
        PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
      }
    }
    class_mat <- matrix(NA, nrow=2^K, ncol=K)
    for(i in 1:(2^K)){class_mat[i,] <- inv_bijectionvector(K, i-1)}
    class_names <- apply(class_mat, 1, paste, collapse="")
    rownames(point_estimates$pis_EAP) <- class_names
    colnames(point_estimates$pis_EAP) <- "pis_EAP"
    rownames(point_estimates$lambdas_EAP) <- paste0("\u03bb", 0:(length(point_estimates$lambdas_EAP)-1))
    res <- list(Model = "DINA_HO",
                Alphas_est = point_estimates$Alphas_est,
                ss_EAP = point_estimates$ss_EAP,
                gs_EAP = point_estimates$gs_EAP,
                thetas_EAP = point_estimates$thetas_EAP,
                pis_EAP = point_estimates$pis_EAP,
                lambdas_EAP = point_estimates$lambdas_EAP,
                
                DIC = HMDCM_fit$DIC,
                PPP_total_scores = PPP_total_scores,
                PPP_item_means = PPP_item_means,
                PPP_item_ORs = PPP_item_ORs)
  }
  
  
  # DINA_HO_RT_sep
  if(object$Model == "DINA_HO_RT_sep"){
    Y_sim <- Dense2Sparse(object$input_data$Response, object$input_data$Test_order, object$input_data$Test_versions)
    L_sim <- Dense2Sparse(object$input_data$Latency, object$input_data$Test_order, object$input_data$Test_versions)
    Q_matrix <- object$input_data$Qs[,,1]
    if(T > 1){
      for(i in 2:T){
        Q_matrix <- rbind(Q_matrix, object$input_data$Qs[,,i])
      }
    }
    point_estimates <- point_estimates_learning(object,"DINA_HO_RT_sep",N,Jt,K,T,alpha_EAP = F)
    HO_RT_sep_fit <- Learning_fit(object,"DINA_HO_RT_sep",Y_sim,Q_matrix,
                              object$input_data$Test_order,object$input_data$Test_versions,
                              Q_examinee=object$input_data$Q_examinee,
                              Latency_array = object$input_data$Latency, G_version = object$input_data$G_version)
    PPP_total_scores <- PPP_total_RTs <- matrix(NA,N,T)
    Y_sim_array <- Sparse2Dense(Y_sim, object$input_data$Test_order, object$input_data$Test_versions)
    L_sim_array <- Sparse2Dense(L_sim, object$input_data$Test_order, object$input_data$Test_versions)
    for(i in 1:N){
      for(t in 1:T){
        f1 <- ecdf(HO_RT_sep_fit$PPs$total_score_PP[i,t,])
        tot_score_it = sum(Y_sim_array[i,,t])
        PPP_total_scores[i,t] = f1(tot_score_it)
        
        f2 <- ecdf(HO_RT_sep_fit$PPs$total_time_PP[i,t,])
        tot_time_it = sum(L_sim_array[i,,t])
        PPP_total_RTs[i,t] = f2(tot_time_it)
      }
    }
    PPP_item_means <- PPP_item_mean_RTs <- numeric(J)
    PPP_item_ORs <- matrix(NA,J,J)
    Y_sim_collapsed <- L_sim_collapsed <- matrix(NA,N,J)
    for(i in 1:N){
      test_i <- object$input_data$Test_versions[i]
      for(t in 1:T){
        t_i = object$input_data$Test_order[test_i,t]
        Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_array[i,,t]
        L_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- L_sim_array[i,,t]
      }
    }
    Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
    for(j in 1:J){
      f1 <- ecdf(HO_RT_sep_fit$PPs$item_mean_PP[j,])
      mean_obs <- mean(Y_sim_collapsed[,j])
      PPP_item_means[j] = f1(mean_obs)
      
      f3 <- ecdf(HO_RT_sep_fit$PPs$RT_mean_PP[j,])
      mean_RT_obs <- mean(L_sim_collapsed[,j])
      PPP_item_mean_RTs[j] = f3(mean_RT_obs)
    }
    for(j in 1:(J-1)){
      for(jp in (j+1):J){
        f2 <- ecdf(HO_RT_sep_fit$PPs$item_OR_PP[j,jp,])
        PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
      }
    }
    response_times_coefficients <- list(as_EAP = point_estimates$as_EAP,
                                        gammas_EAP = point_estimates$gammas_EAP,
                                        taus_EAP = point_estimates$taus_EAP,
                                        phis = point_estimates$phis,
                                        tauvar_EAP = point_estimates$tauvar_EAP)
    class_mat <- matrix(NA, nrow=2^K, ncol=K)
    for(i in 1:(2^K)){class_mat[i,] <- inv_bijectionvector(K, i-1)}
    class_names <- apply(class_mat, 1, paste, collapse="")
    rownames(point_estimates$pis_EAP) <- class_names
    colnames(point_estimates$pis_EAP) <- "pis_EAP"
    rownames(point_estimates$lambdas_EAP) <- paste0("\u03bb", 0:(length(point_estimates$lambdas_EAP)-1))
    res <- list(Model = "DINA_HO_RT_sep",
                Alphas_est = point_estimates$Alphas_est,
                ss_EAP = point_estimates$ss_EAP,
                gs_EAP = point_estimates$gs_EAP,
                thetas_EAP = point_estimates$thetas_EAP,
                pis_EAP = point_estimates$pis_EAP,
                lambdas_EAP = point_estimates$lambdas_EAP,
                response_times_coefficients = response_times_coefficients,
                
                DIC = HO_RT_sep_fit$DIC,
                PPP_total_scores = PPP_total_scores,
                PPP_total_RTs = PPP_total_RTs,
                PPP_item_means = PPP_item_means,
                PPP_item_mean_RTs = PPP_item_mean_RTs,
                PPP_item_ORs = PPP_item_ORs)
  }
  
  
  # DINA_HO_RT_joint
  if(object$Model=="DINA_HO_RT_joint"){
    Y_sim <- Dense2Sparse(object$input_data$Response, object$input_data$Test_order, object$input_data$Test_versions)
    L_sim <- Dense2Sparse(object$input_data$Latency, object$input_data$Test_order, object$input_data$Test_versions)
    Q_matrix <- object$input_data$Qs[,,1]
    if(T > 1){
      for(i in 2:T){
        Q_matrix <- rbind(Q_matrix, object$input_data$Qs[,,i])
      }
    }
    point_estimates <- point_estimates_learning(object,"DINA_HO_RT_joint",N,Jt,K,T,alpha_EAP = F)
    HO_RT_joint_fit <- Learning_fit(object,"DINA_HO_RT_joint",Y_sim,Q_matrix,
                                  object$input_data$Test_order,object$input_data$Test_versions,
                                  Q_examinee=object$input_data$Q_examinee,
                                  Latency_array = object$input_data$Latency, G_version = object$input_data$G_version)
    PPP_total_scores <- PPP_total_RTs <- matrix(NA,N,T)
    Y_sim_array <- Sparse2Dense(Y_sim, object$input_data$Test_order, object$input_data$Test_versions)
    L_sim_array <- Sparse2Dense(L_sim, object$input_data$Test_order, object$input_data$Test_versions)
    for(i in 1:N){
      for(t in 1:T){
        f1 <- ecdf(HO_RT_joint_fit$PPs$total_score_PP[i,t,])
        tot_score_it = sum(Y_sim_array[i,,t])
        PPP_total_scores[i,t] = f1(tot_score_it)
        
        f2 <- ecdf(HO_RT_joint_fit$PPs$total_time_PP[i,t,])
        tot_time_it = sum(L_sim_array[i,,t])
        PPP_total_RTs[i,t] = f2(tot_time_it)
      }
    }
    PPP_item_means <- PPP_item_mean_RTs <- numeric(J)
    PPP_item_ORs <- matrix(NA,J,J)
    Y_sim_collapsed <- L_sim_collapsed <- matrix(NA,N,J)
    for(i in 1:N){
      test_i <- object$input_data$Test_versions[i]
      for(t in 1:T){
        t_i = object$input_data$Test_order[test_i,t]
        Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_array[i,,t]
        L_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- L_sim_array[i,,t]
      }
    }
    Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
    for(j in 1:J){
      f1 <- ecdf(HO_RT_joint_fit$PPs$item_mean_PP[j,])
      mean_obs <- mean(Y_sim_collapsed[,j])
      PPP_item_means[j] = f1(mean_obs)
      
      f3 <- ecdf(HO_RT_joint_fit$PPs$RT_mean_PP[j,])
      mean_RT_obs <- mean(L_sim_collapsed[,j])
      PPP_item_mean_RTs[j] = f3(mean_RT_obs)
    }
    for(j in 1:(J-1)){
      for(jp in (j+1):J){
        f2 <- ecdf(HO_RT_joint_fit$PPs$item_OR_PP[j,jp,])
        PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
      }
    }
    response_times_coefficients <- list(as_EAP = point_estimates$as_EAP,
                                        gammas_EAP = point_estimates$gammas_EAP,
                                        taus_EAP = point_estimates$taus_EAP,
                                        phis = point_estimates$phis,
                                        Sigs_EAP = point_estimates$Sigs_EAP)
    class_mat <- matrix(NA, nrow=2^K, ncol=K)
    for(i in 1:(2^K)){class_mat[i,] <- inv_bijectionvector(K, i-1)}
    class_names <- apply(class_mat, 1, paste, collapse="")
    rownames(point_estimates$pis_EAP) <- class_names
    colnames(point_estimates$pis_EAP) <- "pis_EAP"
    rownames(point_estimates$lambdas_EAP) <- paste0("\u03bb", 0:(length(point_estimates$lambdas_EAP)-1))
    res <- list(Model = "DINA_HO_RT_joint",
                Alphas_est = point_estimates$Alphas_est,
                ss_EAP = point_estimates$ss_EAP,
                gs_EAP = point_estimates$gs_EAP,
                thetas_EAP = point_estimates$thetas_EAP,
                pis_EAP = point_estimates$pis_EAP,
                lambdas_EAP = point_estimates$lambdas_EAP,
                response_times_coefficients = response_times_coefficients,
                
                DIC = HO_RT_joint_fit$DIC,
                PPP_total_scores = PPP_total_scores,
                PPP_total_RTs = PPP_total_RTs,
                PPP_item_means = PPP_item_means,
                PPP_item_mean_RTs = PPP_item_mean_RTs,
                PPP_item_ORs = PPP_item_ORs)
  }
  
  
  # rRUM_indept
  if(object$Model=="rRUM_indept"){
    Y_sim <- Dense2Sparse(object$input_data$Response, object$input_data$Test_order, object$input_data$Test_versions)
    Q_matrix <- object$input_data$Qs[,,1]
    if(T > 1){
      for(i in 2:T){
        Q_matrix <- rbind(Q_matrix, object$input_data$Qs[,,i])
      }
    }
    point_estimates = point_estimates_learning(object,"rRUM_indept",N,Jt,K,T,alpha_EAP = T)
    rRUM_indept_fit <- Learning_fit(object,"rRUM_indept",Y_sim,Q_matrix,
                                    object$input_data$Test_order,object$input_data$Test_versions,
                                    R=object$input_data$R)
    PPP_total_scores <- matrix(NA,N,T)
    Y_sim_array <- Sparse2Dense(Y_sim, object$input_data$Test_order, object$input_data$Test_versions)
    for(i in 1:N){
      for(t in 1:T){
        f <- stats::ecdf(rRUM_indept_fit$PPs$total_score_PP[i,t,])
        tot_it = sum(Y_sim_array[i,,t])
        PPP_total_scores[i,t] = f(tot_it)
      }
    }
    PPP_item_means <- numeric(J)
    PPP_item_ORs <- matrix(NA,J,J)
    Y_sim_collapsed <- matrix(NA,N,J)
    for(i in 1:N){
      test_i <- object$input_data$Test_versions[i]
      for(t in 1:T){
        t_i = object$input_data$Test_order[test_i,t]
        Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_array[i,,t]
      }
    }
    Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
    for(j in 1:J){
      f1 <- stats::ecdf(rRUM_indept_fit$PPs$item_mean_PP[j,])
      mean_obs <- mean(Y_sim_collapsed[,j])
      PPP_item_means[j] = f1(mean_obs)
    }
    for(j in 1:(J-1)){
      for(jp in (j+1):J){
        f2 <- stats::ecdf(rRUM_indept_fit$PPs$item_OR_PP[j,jp,])
        PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
      }
    }
    class_mat <- matrix(NA, nrow=2^K, ncol=K)
    for(i in 1:(2^K)){class_mat[i,] <- inv_bijectionvector(K, i-1)}
    class_names <- apply(class_mat, 1, paste, collapse="")
    rownames(point_estimates$pis_EAP) <- class_names
    colnames(point_estimates$pis_EAP) <- "pis_EAP"
    rownames(point_estimates$taus_EAP) <- paste0("\u03c4", 1:(length(point_estimates$taus_EAP)))
    res <- list(Model = "rRUM_indept",
                Alphas_est = point_estimates$Alphas_est,
                taus_EAP = point_estimates$taus_EAP,
                r_stars_EAP = point_estimates$r_stars_EAP,
                pi_stars_EAP = point_estimates$pi_stars_EAP,
                pis_EAP = point_estimates$pis_EAP,
                
                DIC = rRUM_indept_fit$DIC,
                PPP_total_scores = PPP_total_scores,
                PPP_item_means = PPP_item_means,
                PPP_item_ORs = PPP_item_ORs)

  }
  
  # NIDA_indept
  if(object$Model=="NIDA_indept"){
    Y_sim <- Dense2Sparse(object$input_data$Response, object$input_data$Test_order, object$input_data$Test_versions)
    Q_matrix <- object$input_data$Qs[,,1]
    if(T > 1){
      for(i in 2:T){
        Q_matrix <- rbind(Q_matrix, object$input_data$Qs[,,i])
      }
    }
    point_estimates = point_estimates_learning(object,"NIDA_indept",N,Jt,K,T,alpha_EAP = T)
    NIDA_indept_fit <- Learning_fit(object,"NIDA_indept",Y_sim,Q_matrix,
                                    object$input_data$Test_order,object$input_data$Test_versions,
                                    R=object$input_data$R)
    PPP_total_scores <- matrix(NA,N,T)
    Y_sim_array <- Sparse2Dense(Y_sim, object$input_data$Test_order, object$input_data$Test_versions)
    for(i in 1:N){
      for(t in 1:T){
        f <- stats::ecdf(NIDA_indept_fit$PPs$total_score_PP[i,t,])
        tot_it = sum(Y_sim_array[i,,t])
        PPP_total_scores[i,t] = f(tot_it)
      }
    }
    PPP_item_means <- numeric(J)
    PPP_item_ORs <- matrix(NA,J,J)
    Y_sim_collapsed <- matrix(NA,N,J)
    for(i in 1:N){
      test_i <- object$input_data$Test_versions[i]
      for(t in 1:T){
        t_i = object$input_data$Test_order[test_i,t]
        Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_array[i,,t]
      }
    }
    Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
    for(j in 1:J){
      f1 <- stats::ecdf(NIDA_indept_fit$PPs$item_mean_PP[j,])
      mean_obs <- mean(Y_sim_collapsed[,j])
      PPP_item_means[j] = f1(mean_obs)
    }
    for(j in 1:(J-1)){
      for(jp in (j+1):J){
        f2 <- stats::ecdf(NIDA_indept_fit$PPs$item_OR_PP[j,jp,])
        PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
      }
    }
    class_mat <- matrix(NA, nrow=2^K, ncol=K)
    for(i in 1:(2^K)){class_mat[i,] <- inv_bijectionvector(K, i-1)}
    class_names <- apply(class_mat, 1, paste, collapse="")
    rownames(point_estimates$pis_EAP) <- class_names
    colnames(point_estimates$pis_EAP) <- "pis_EAP"
    rownames(point_estimates$taus_EAP) <- paste0("\u03c4", 1:(length(point_estimates$taus_EAP)))
    res <- list(Model = "NIDA_indept",
                Alphas_est = point_estimates$Alphas_est,
                ss_EAP = point_estimates$ss_EAP,
                gs_EAP = point_estimates$gs_EAP,
                taus_EAP = point_estimates$taus_EAP,
                pis_EAP = point_estimates$pis_EAP,

                DIC = NIDA_indept_fit$DIC,
                PPP_total_scores = PPP_total_scores,
                PPP_item_means = PPP_item_means,
                PPP_item_ORs = PPP_item_ORs)
  }
  
  # DINA_FOHM
  if(object$Model=="DINA_FOHM"){
    Y_sim <- Dense2Sparse(object$input_data$Response, object$input_data$Test_order, object$input_data$Test_versions)
    Q_matrix <- object$input_data$Qs[,,1]
    if(T > 1){
      for(i in 2:T){
        Q_matrix <- rbind(Q_matrix, object$input_data$Qs[,,i])
      }
    }
    point_estimates = point_estimates_learning(object,"DINA_FOHM",N,Jt,K,T,alpha_EAP = T)
    DINA_FOHM_fit <- Learning_fit(object,"DINA_FOHM",Y_sim,Q_matrix,
                             object$input_data$Test_order,object$input_data$Test_versions)
    PPP_total_scores <- matrix(NA,N,T)
    Y_sim_array <- Sparse2Dense(Y_sim, object$input_data$Test_order, object$input_data$Test_versions)
    for(i in 1:N){
      for(t in 1:T){
        f <- stats::ecdf(DINA_FOHM_fit$PPs$total_score_PP[i,t,])
        tot_it = sum(Y_sim_array[i,,t])
        PPP_total_scores[i,t] = f(tot_it)
      }
    }
    PPP_item_means <- numeric(J)
    PPP_item_ORs <- matrix(NA,J,J)
    Y_sim_collapsed <- matrix(NA,N,J)
    for(i in 1:N){
      test_i <- object$input_data$Test_versions[i]
      for(t in 1:T){
        t_i = object$input_data$Test_order[test_i,t]
        Y_sim_collapsed[i,(Jt*(t_i-1)+1):(Jt*t_i)] <- Y_sim_array[i,,t]
      }
    }
    Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
    for(j in 1:J){
      f1 <- stats::ecdf(DINA_FOHM_fit$PPs$item_mean_PP[j,])
      mean_obs <- mean(Y_sim_collapsed[,j])
      PPP_item_means[j] = f1(mean_obs)
    }
    for(j in 1:(J-1)){
      for(jp in (j+1):J){
        f2 <- stats::ecdf(DINA_FOHM_fit$PPs$item_OR_PP[j,jp,])
        PPP_item_ORs[j,jp] <- f2(Observed_ORs[j,jp])
      }
    }
    class_mat <- matrix(NA, nrow=2^K, ncol=K)
    for(i in 1:(2^K)){class_mat[i,] <- inv_bijectionvector(K, i-1)}
    class_names <- apply(class_mat, 1, paste, collapse="")
    rownames(point_estimates$pis_EAP) <- class_names
    colnames(point_estimates$pis_EAP) <- "pis_EAP"
    res <- list(Model = "DINA_FOHM",
                Alphas_est = point_estimates$Alphas_est,
                ss_EAP = point_estimates$ss_EAP,
                gs_EAP = point_estimates$gs_EAP,
                omegas_EAP = point_estimates$omegas_EAP,
                pis_EAP = point_estimates$pis_EAP,
                
                DIC = DINA_FOHM_fit$DIC,
                PPP_total_scores = PPP_total_scores,
                PPP_item_means = PPP_item_means,
                PPP_item_ORs = PPP_item_ORs)
  }
  
  class(res) <- "summary.hmcdm"
  return(res)
}

#' @param x an object of class "`hmcdm.summary`".
#' @rdname summary.hmcdm
#' @export
print.summary.hmcdm <- function(x, ...){
  digits <- max(3, getOption("digits") - 3)
  
  cat("\nModel:",x$Model,"\n")
  
  cat("\nItem Parameters:\n")
  K <- log(nrow(x$pis_EAP), base=2)
  if(x$Model != "rRUM_indept"){
    Item_parameters <- cbind(x$ss_EAP,x$gs_EAP)
    colnames(Item_parameters) <- c("ss_EAP", "gs_EAP")}
  if(x$Model == "rRUM_indept"){
    Item_parameters <- cbind(x$r_stars_EAP, x$pi_stars_EAP)
    colnames(Item_parameters) <- c(paste0("r_stars",1:K,"_EAP"), "pi_stars_EAP")}
  rownames(Item_parameters) <- rep("",nrow(Item_parameters))
  print(head(Item_parameters, 5), digits=digits)
  if(nrow(Item_parameters)>5){cat("   ...", nrow(Item_parameters)-5, "more items\n")}
  
  cat("\nTransition Parameters:\n")
  if(x$Model=="DINA_HO" || x$Model=="DINA_HO_RT_sep" || x$Model=="DINA_HO_RT_joint"){
    Transition_parameters <- x$lambdas_EAP
    rownames(Transition_parameters) <- paste0("\u03bb", 0:(length(Transition_parameters)-1))
    colnames(Transition_parameters) <- "lambdas_EAP"
  }
  if(x$Model=="NIDA_indept" || x$Model=="rRUM_indept"){
    Transition_parameters <- x$taus_EAP
    rownames(Transition_parameters) <- paste0("\u03c4", 1:(length(Transition_parameters)))
    colnames(Transition_parameters) <- "taus_EAP"
  }
  if(x$Model=="DINA_FOHM"){
    Transition_parameters <- x$omegas_EAP[1,]
    Transition_parameters_name <- "omegas_EAP"
  }
  print(Transition_parameters, digits=digits)
  if(x$Model=="DINA_FOHM"){cat("   ...", length(Transition_parameters)-1, "more rows\n")}
  
  cat("\nClass Probabilities:\n")
  class_mat <- matrix(NA, nrow=2^K, ncol=K)
  for(i in 1:(2^K)){class_mat[i,] <- inv_bijectionvector(K, i-1)}
  class_names <- apply(class_mat, 1, paste, collapse="")
  rownames(x$pis_EAP) <- class_names
  colnames(x$pis_EAP) <- "pis_EAP"
  print(head(x$pis_EAP, 5), digits=digits)
  if(length(x$pis_EAP)>5){cat("   ...", length(x$pis_EAP)-5, "more classes\n")}
  
  cat("\nDeviance Information Criterion (DIC):", x$DIC["DIC","Total"],"\n")
  
  cat("\nPosterior Predictive P-value (PPP):")
  cat("\nM1:", formatC(mean(x$PPP_item_means), digits=digits))
  cat("\nM2:", formatC(mean(upper.tri(x$PPP_item_ORs)), digits=digits))
  cat("\ntotal scores: ", formatC(mean(x$PPP_total_scores), digits=digits))

  invisible(x)
}




