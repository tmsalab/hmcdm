#' Observed response times list
#' 
#' This data set contains the observed latencies of responses of all subjects to all questions in the Spatial Rotation 
#' Learning Program.
#' @format A list of length 5 (number of time points). Each element of the list is an N-by-Jt matrix, containing the
#' subjects' response times in seconds to each item at that time point.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"L_real_list"


#' List of Q matrices
#' 
#' This data set contains the Q matrices of the items in the Spatial Rotation Learning Program.
#' @format A list of length 5 (number of item blocks). Each element of the list is a Jt-by-K matrix, containing the 
#' item-skill relationship of items in the corresponding block.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Q_list"


#' Array of Q matrices
#' 
#' This array contains the Q matrices of the items in the Spatial Rotation Learning Program.
#' @format An array of dimensions 10-by-4-by-5. Each slice of the array is a Jt-by-K matrix, containing the 
#' item-skill relationship of items in the corresponding block.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Qs"


#' List of Q-matrices for each examinee.
#' 
#' This data set contains the Q matrices for each subject in the Spatial Rotation Learning Program.
#' @format A list of length 350. Each element of the list is a 50x4 matrix, containing the Q matrix of all items 
#' administered across all time points to the examinee, in the order of administration.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Q_examinee"



#' Subjects' test version
#' 
#' This data set contains each subject's test version in the Spatial Rotation Learning Program.
#' @format A vector of length 350, containing each subject's test version ranging from 1 to 5.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Test_versions"


#' Observed response accuracy list
#' 
#' This data set contains each subject's observed response accuracy (0/1) at all time points in the Spatial 
#' Rotation Learning Program.
#' @format A list of length 5 (number of time points). Each element of the list is an N-by-Jt matrix, containing the
#' subjects' response accuracy to each item at that time point.
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
"Y_real_list"


#' Test block ordering of each test version
#' 
#' This data set contains the item block ordering of each version of the test.
#' @format A 5x5 matrix, each row is the order of item blocks (as in Qs and Q_list) for that test version. 
#' For example, the first row is the order of item block administration (1-2-3-4-5) to subjects with test
#' version 1. 
#' @source Spatial Rotation Learning Experiment at UIUC between Fall 2015 and Spring 2016.
#' @author Shiyu Wang, Yan Yang, Jeff Douglas, and Steve Culpepper
'test_order'