#' @title Train model again and evaluate the performance of the model.
#' @description Train model again in all training dataset and evaluate the performance of the model in the test dataset.
#' @param data_train \code{data.frame} All training dataset.
#' @param data_test  \code{data.frame} All test dataset.
#' @param lambda   \code{numeric} The best lambda that maximizes auc.
#' @param maxIterNum  \code{numeric} Maximum number of iterations.
#' @param step \code{numeric} Gradient descent stepsize.
#'
#' @return final_res \code{list} Contains the regression coefficients,prediction probability and category,confusion matrix,recall and precision.
#' @export
#' @importFrom  tibble tibble
#' @importFrom dplyr mutate
#' @importFrom yardstick roc_auc
#' @importFrom yardstick conf_mat
#' @importFrom yardstick recall
#' @importFrom yardstick precision
#' @importFrom  purrr as_vector
#' @importFrom  magrittr  %>%
#' @importFrom utils data
#' @examples
#' set.seed(2020)
#' dat = iris %>% dplyr::filter(Species != "setosa")
#' dat$Species = ifelse(dat$Species == "versicolor",0,1)
#' sam_id = sample(1:nrow(dat),nrow(dat)*0.75,replace = FALSE)
#' df_train = dat[sam_id,]
#' df_test = dat[-sam_id,]
#' final_test(data_train = df_train,data_test = df_test,
#'             lambda = 0.1, maxIterNum = 100000,step = 0.0005)



final_test = function(data_train = data_train,data_test = data_test,
                      lambda = lambda, maxIterNum = 100000,step = 0.0005){
  V3 = NULL
  Y_pred_prob_1 = NULL

  X_test = cbind(rep(1,nrow(data_test)),
                 data_test[,-ncol(data_test)]) %>%
    as.matrix()
  Y_test = data_test[,ncol(data_test)] %>% as_vector()


  # 带入交叉验证选出最好的lambda在训练集训练模型
  X_train = cbind(rep(1,nrow(data_train)),
                  data_train[,-ncol(data_train)]) %>%
    as.matrix()
  Y_train = data_train[,ncol(data_train)] %>% as_vector()
  maxIterNum = maxIterNum         # 最大迭代数
  step = step                     # 学习步长
  W = rep(0, ncol(X_train))       # 初始参数值
  m = nrow(X_train)
  sigmoid = function(z) { 1 / (1 + exp(-z))}     # sigmoid函数
  for (i in 1:maxIterNum){
    grad = t(X_train) %*% (sigmoid(X_train %*% W)-Y_train)# 梯度
    if (sqrt(as.numeric(t(grad) %*% grad)) < 1e-8){
      print(sprintf('iter times=%d', i))
      break
    }
    W = W - grad * step - step*lambda*sign(W)
  }
  W
  # print(W)
  hfunc <- function(a) {if (a > 0.5) return(1) else return (0)}
  Y_pred_class = apply(sigmoid(X_test %*% W), 1, hfunc)
  Y_pred_prob = sigmoid(X_test %*% W)
  tmp = cbind(Y_test,cbind(Y_pred_class,Y_pred_prob)) %>%
    as.data.frame() %>%
    mutate(Y_test = factor(Y_test),
           Y_pred_class = factor(Y_pred_class),
           Y_pred_prob_1 = 1-V3)
  names(tmp) = c("Y_test","Y_pred_class","Y_pred_prob_0","Y_pred_prob_1")
  tmp = tibble(tmp)
  tmp
  # print(tmp)
  auc = roc_auc(tmp,Y_test,Y_pred_prob_1)
  auc
  confusion_matrix = conf_mat(tmp, Y_test,Y_pred_class)
  confusion_matrix
  recall_ = recall(tmp, Y_test,Y_pred_class,event_level = "second")
  recall_
  precision_ = precision(tmp, Y_test,Y_pred_class,event_level = "second")
  precision_
  # print(auc)
  final_res = list(param = W,AUC = auc, pred_outcome = tmp,
                   confusion_matrix = confusion_matrix,
                   recall = recall_,precision = precision_)
  return(final_res)
}
