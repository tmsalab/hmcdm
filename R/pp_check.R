#' @title Graphical posterior predictive checks for hidden Markov cognitive diagnosis model
#' @description `pp_check` method for class `hmcdm`.
#' @param object a fitted model object of class "`hmcdm`".
#' @param plotfun A character string naming the type of plot. The list of available 
#' plot functions include `"dens_overlay"`, `"hist"`, `"stat_2d"`, `"scatter_avg"`, `"error_scatter_avg"`.
#' The default function is `"dens_overlay"`.
#' @param type A character string naming the statistic to be used for obtaining posterior predictive distribution plot. 
#' The list of available types include `"total_score"`, `"item_mean"`, `"item_OR"`, `"latency_mean"`, and `"latency_total"`. The default type is `"total_score"` which examines total scores of subjects. 
#' Type `"item_mean"` is related to the first order moment and examines mean scores of all the items included in the test. 
#' Type `"item_OR"` is related to the second order moment and examines odds ratios of all item pairs.
#' Types `"latency_mean"` and `"total_latency"` are available only for `hmcdm` objects that include item response time information (i.e., `hmcdm` object fitted with "`DINA_HO_RT`" model).
#' @return Plots for checking the posterior predictive distributions. The default `Plotfun` `"dens_overlay"` plots density of each dataset are overlaid with the distribution of the observed values.
#' @seealso 
#' [bayesplot::ppc_dens_overlay()]
#' [bayesplot::ppc_stat()]
#' [bayesplot::ppc_stat_2d()]
#' [bayesplot::ppc_scatter_avg()]
#' [bayesplot::ppc_error_scatter_avg()]
#' @references 
#' Zhang, S., Douglas, J. A., Wang, S. & Culpepper, S. A. (2019) <doi:10.1007/978-3-030-05584-4_24>
#' @examples
#' \donttest{
#' output_FOHM = hmcdm(Y_real_array,Q_matrix,"DINA_FOHM",Design_array,1000,500)
#' library(bayesplot)
#' pp_check(output_FOHM)
#' pp_check(output_FOHM, plotfun="hist", type="item_mean")
#' }
#' @export
pp_check.hmcdm <- function(object,plotfun="dens_overlay",type="total_score"){
  N <- dim(object$input_data$Response)[1]
  L <- dim(object$input_data$Response)[3]
  J <- dim(object$input_data$Response)[2]
  
  Y_sim <- object$input_data$Response
  Design_array <- object$input_data$Design_array
  Q_matrix <- object$input_data$Q_matrix
  Q_examinee <- object$input_data$Q_examinee
  
  Y_sim_collapsed <- matrix(NA,N,J)
  for(i in 1:N){
    for(t in 1:L){
      block_it <- which(!is.na(Design_array[i,,t]))
      Y_sim_collapsed[i,block_it] <- Y_sim[i,block_it,t]
    }
  }
  
  if(object$Model == "DINA_HO"){
    object_fit <- Learning_fit_g(object, "DINA_HO", Y_sim,Q_matrix, 
                          Design_array, Q_examinee)
  }
  if(object$Model == "DINA_HO_RT_sep" | object$Model == "DINA_HO_RT_joint"){
    L_sim <- object$input_data$Latency
    object_fit <- Learning_fit_g(object,object$Model,Y_sim,Q_matrix,
                                  Design_array,
                                  Q_examinee=object$input_data$Q_examinee,
                                  Latency_array = L_sim, G_version = object$input_data$G_version)
  }
  if(object$Model == "rRUM_indept" | object$Model == "NIDA_indept"){
    object_fit <- Learning_fit_g(object,object$Model,Y_sim,Q_matrix,
                                    Design_array,
                                    R=object$input_data$R)
  }
  if(object$Model == "DINA_FOHM"){
    object_fit <- Learning_fit_g(object,"DINA_FOHM",Y_sim,Q_matrix,
                          Design_array)
  }
  
  if(type=="total_score"){
    ## total score
    total_score_obs <- matrix(NA, N, L)
    for(t in 1:L){
      total_score_obs[,t] <- rowSums(object$input_data$Response[,,t], na.rm=TRUE)
    }
    obs <- rowSums(total_score_obs)
    pp <- apply(object_fit$PPs$total_score_PP,1,colSums)
  }
  if(type=="item_mean"){
    ## Item means
    obs <- rep(NA, J)
    for(j in 1:J){
      obs[j] <- mean(Y_sim_collapsed[,j], na.rm=TRUE)
    }
    pp <- t(object_fit$PPs$item_mean_PP)
  }
  if(type=="item_OR"){
    ## Item log odds ratio
    Observed_ORs <- OddsRatio(N,J,Y_sim_collapsed)
    ORs_obs <- Observed_ORs[upper.tri(Observed_ORs)]
    obs <- log(ORs_obs)
    CL <- object$chain_length-object$burn_in
    ORs_pp <- matrix(NA, nrow=CL, ncol=length(ORs_obs))
    for(cl in 1:CL){
      ORs_pp[cl,] <- object_fit$PPs$item_OR_PP[,,cl][upper.tri(object_fit$PPs$item_OR_PP[,,cl])]
    }
    pp <- log(ORs_pp)
  }
  if(type=="latency_mean"){
    L_sim_collapsed <- matrix(NA,N,J)
    for(i in 1:N){
      for(t in 1:L){
        block_it <- which(!is.na(Design_array[i,,t]))
        L_sim_collapsed[i,block_it] <- L_sim[i,block_it,t]
      }
    }
    ## latency means
    obs <- rep(NA, J)
    for(j in 1:J){
      obs[j] <- mean(L_sim_collapsed[,j], na.rm=TRUE)
    }
    pp <- t(object_fit$PPs$RT_mean_PP)
  }
  if(type=="total_latency"){
    ## total score
    total_latency_obs <- matrix(NA, N, L)
    for(t in 1:L){
      total_latency_obs[,t] <- rowSums(object$input_data$Latency[,,t], na.rm=TRUE)
    }
    obs <- rowSums(total_latency_obs)
    pp <- apply(object_fit$PPs$total_time_PP,1,colSums)
  }
  
  if(plotfun=="dens_overlay"){# Compare distribution of y to distributions of multiple yrep datasets
    bayesplot::color_scheme_set("red")
    result <- bayesplot::ppc_dens_overlay(y=obs, yrep=pp)
  }
  if(plotfun=="hist"){# Check histograms of test statistics
    bayesplot::color_scheme_set("blue")
    result <- bayesplot::ppc_stat(y=obs, yrep=pp)
  }
  if(plotfun=="stat_2d"){# Scatterplot of two test statistics
    bayesplot::color_scheme_set("blue")
    result <- bayesplot::ppc_stat_2d(y=obs, yrep=pp)
  }
  if(plotfun=="scatter_avg"){  # Scatterplot of y vs. average yrep
    bayesplot::color_scheme_set("blue")
    result <- bayesplot::ppc_scatter_avg(y = obs, yrep = pp)
  }
  if(plotfun=="error_scatter_avg"){# predictive errors
    bayesplot::color_scheme_set("blue")
    result <- bayesplot::ppc_error_scatter_avg(y = obs, yrep = pp)
  }
  
  return(result)
}



