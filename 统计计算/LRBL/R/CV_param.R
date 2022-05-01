#' @title Cross validation to choose lambda
#' @description  Cross validation to choose the best lambda that maximizes auc.
#' @param dataset \code{data.frame} The dataset you want to split into k-folds and the label column of your dataset should be in the last column.
#' @param dat_list \code{list} A list contains the line numbers of every fold.
#' @param penalty \code{vector} A vector contains a series of lambda to regularize.
#' @param k \code{numeric} The k-folds you want to split your training data into.
#' @param n \code{numeric} The number of lambda you have.
#'
#' @return cv_out \code{list} Contains the mean of auc of k models trained by each lambda and the lambda that maximizes auc.
#' @export
#' @examples
#' dat = iris %>% dplyr::filter(Species != "setosa")
#' dat$Species = ifelse(dat$Species == "versicolor",0,1)
#' penalty = 10^seq(-4, -1, length.out = 10)
#' cv_dat = CV_split(10,100,2020)
#' CV_param(dataset = dat,dat_list = cv_dat,penalty = penalty,k = 10 , n = 10)



CV_param = function(dataset = data,dat_list = dat_list,
                    penalty = penalty,k = k,n = n ){
  V3 = NULL
  Y_val_pred_prob_1 = NULL
  k = k
  n = n
  # k = 5
  # n = 10
  dataset = dataset
  cv_list = dat_list
  # dataset = df_train
  cv.auc = matrix(NA,k,n)
  # n个超参数
  for (i in 1:n){
    lambda = penalty[i]
    print(i)
    # k折交叉验证，轮流做验证集
    for (j in 1:k){
      # 划分训练验证
      # j = 1
      # dataset = df_train
      train_dat = dataset[-cv_list[[j]],]
      validation_dat = dataset[cv_list[[j]],]
      X_val = cbind(rep(1,nrow(validation_dat)),
                    validation_dat[,-ncol(validation_dat)]) %>%
        as.matrix()

      Y_val = validation_dat[ncol(validation_dat)] %>% as_vector()

      # k-1折训练
      X_train = cbind(rep(1,nrow(train_dat)),
                      train_dat[,-ncol(train_dat)]) %>%
        as.matrix()

      Y_train = train_dat[,ncol(train_dat)] %>% as_vector()
      maxIterNum = 10000              # 最大迭代数
      step = 0.01                    # 学习步长
      W = rep(0, ncol(X_train))       # 初始参数值
      m = nrow(X_train)
      sigmoid = function(z) { 1 / (1 + exp(-z))}  # sigmoid函数
      for (l  in 1:maxIterNum){
        grad = t(X_train) %*% (sigmoid(X_train %*% W)-Y_train)# 梯度
        if (sqrt(as.numeric(t(grad) %*% grad)) < 1e-8){
          print(sprintf('iter times=%d', l))
          break
        }
        W = W - grad * step - step*lambda*sign(W)
      }
      W

      # 1折验证
      Y_val_pred_prob = sigmoid(X_val %*% round(W,3))
      hfunc = function(a) {if (a > 0.5) return(1) else return (0)}
      Y_val_pred_class = apply(sigmoid(X_val  %*%
                                         round(W,3)), 1, hfunc)

      val_bind = cbind(Y_val,
                       cbind(Y_val_pred_class,Y_val_pred_prob)) %>%
        as.data.frame() %>%
        mutate(Y_val = factor(Y_val),
               Y_pred_prob_1 = 1-V3)
      names(val_bind) = c("Y_val","Y_val_pred_class",
                          "Y_val_pred_prob_0", "Y_val_pred_prob_1")

      val_bind = tibble(val_bind)
      # val_bind
      # print(val_bind)
      auc = roc_auc(val_bind,Y_val,Y_val_pred_prob_1)
      # auc
      cv.auc[j,i] = auc[[3]]
    }
  }
  cv.auc_ = apply(cv.auc, 2, mean)
  # return(penalty[which.max(cv.auc_)])
  cv_out = list(cv_auc = cv.auc_,
                best_penalty = penalty[which.max(cv.auc_)])
}
