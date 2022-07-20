## code to prepare `DATASET` dataset goes here

N <- 350
J <- 50
TT <- 5


# a. Convert 'Y_real_list' to an (N*J*T) sparse array.
dim(Y_real_list[[1]])
length(Y_real_list)
Y_real_array <- array(NA, dim=c(N, J, TT))
for(i in 1:N){
  test_version <- Test_versions[[i]]
  for(j in 1:TT){
    test_index <- test_order[test_version,j]
    item_indices <- (10*(test_index-1)+1):(10*test_index)
    Y_real_array[i,item_indices,j] <- Y_real_list[[j]][i,]
  }
}
usethis::use_data(Y_real_array, overwrite=T)




# b. Convert 'L_real_list' to an (N*J*T) sparse array.
dim(L_real_list[[1]])
L_real_array <- array(NA, dim=c(N, J, TT))
for(i in 1:N){
  test_version <- Test_versions[[i]]
  for(j in 1:TT){
    test_index <- test_order[test_version,j]
    item_indices <- (10*(test_index-1)+1):(10*test_index)
    L_real_array[i,item_indices,j] <- L_real_list[[j]][i,]
  }
}
usethis::use_data(L_real_array, overwrite=T)




# c. Convert `Q_list` to an J*K matrix
length(Q_list)
Q_matrix <- rbind(Q_list[[1]],
                 Q_list[[2]],
                 Q_list[[3]],
                 Q_list[[4]],
                 Q_list[[5]])
usethis::use_data(Q_matrix, overwrite=TRUE)


## Design matrix (N*J*T)
Design_array <- array(NA, dim=c(N, J, TT))
for(i in 1:N){
  test_version <- Test_versions[[i]]
  for(t in 1:TT){
    test_index <- test_order[test_version,t]
    item_indices <- (10*(test_index-1)+1):(10*test_index)
    Design_array[i,item_indices,t] <- 1
  }
}
usethis::use_data(Design_array, overwrite=TRUE)

Test_order <- test_order
usethis::use_data(Test_order, overwrite=TRUE)


test_order
Test_versions
Qs
Y_real_list
Q_examinee
length(Q_examinee)
dim(Q_examinee[[1]])





