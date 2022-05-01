#' @title Data split
#' @description Line numbers of dataset split to k-folds randomly.
#' @param k \code{numeric} The k-folds you want to split your training data into.
#' @param datasize \code{numeric}  Your training data size.
#' @param seed \code{numeric} A seed to sample.
#'
#' @return data_list \code{list} Contains line numbers of k-folds data.
#' @export
#'
#' @examples CV_split(10,100,2020)
CV_split = function(k, datasize, seed){
  data_list = list()
  set.seed(seed)
  n = rep(1:k,ceiling(datasize/k))[1:datasize]
  tmp = sample(n,datasize)
  x = 1:k
  dataseq = 1:datasize
  data_list = lapply(x, function(x) dataseq[tmp==x])
  return(data_list)
}
