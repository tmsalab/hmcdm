#' Observed response accuracy array
#' 
#' `Y_real_array` contains each subject's observed response accuracy (0/1) at all time points in the Spatial 
#' Rotation Learning Program.
#' @format An array of dimensions N-by-J-by-L. Each slice of the array is an N-by-J matrix, containing the
#' subjects' response accuracy to each item at time point l.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Y_real_array"


#' Observed response times array
#' 
#' `L_real_array` contains the observed latencies of responses of all subjects to all questions in the Spatial Rotation 
#' Learning Program.
#' @format An array of dimensions N-by-J-by-L. Each slice of the array is an N-by-J matrix, containing the
#' subjects' response times in seconds to each item at time point l.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"L_real_array"


#' Q-matrix
#' 
#' `Q_matrix` contains the Q matrix of the items in the Spatial Rotation Learning Program.
#' @format A J-by-K matrix, indicating the item-skill relationship.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Q_matrix"


#' Subjects' test version
#' 
#' `Test_versions` contains each subject's test module in the Spatial Rotation Learning Program.
#' @format A vector of length N, containing each subject's assigned test module.
#' @details The data object `"Test_versions"` contains a vector of length N indicating the test module assigned to each subject. 
#' Each test module consists of multiple item blocks with different orders over L time points. 
#' The order of item blocks corresponding to each test module is presented in the data object `"Test_order"`.
#' @seealso [`Test_order`]
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Test_versions"


#' Test block ordering of each test version
#' 
#' `Test_order` contains the item block ordering corresponding to each test module.
#' @format A L-by-L matrix, each row is the order of item blocks for that test version. 
#' @details Each row represents the test module number and shows the order of item blocks administered to a subject with the test module. 
#' For example, the first row is the order of item block administration (1-2-3-4-5) to subjects with test
#' module 1. 
#' @seealso [`Test_versions`]
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
'Test_order'

