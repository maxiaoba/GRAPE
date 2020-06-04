library(xgboost)
library(rpart)

setwd("/Users/daisyding/Desktop/2020_Spring/MDI_GNN/uci/data_with_missing/0.3/")

datasets = c('concrete', 'energy', 'housing', 'kin8nm',
          'naval', 'power', 'protein', 'wine', 'yacht')
seed_list = seq(5)-1

mae_list_xgb = matrix(0, nrow=length(datasets), ncol=length(seed_list))
mae_list_tree = matrix(0, nrow=length(datasets), ncol=length(seed_list))

for (i in seq(datasets)){
  for (j in seq(seed_list)){
    data = datasets[i]
    seed = seed_list[j]
    
    x_train = read.csv(paste0("./",data,"/",seed,"/train_X_missing.txt"), sep=",")
    y_train = read.csv(paste0("./",data,"/",seed,"/train_y.txt"))
    x_test = read.csv(paste0("./",data,"/",seed,"/test_X_missing.txt"), sep=",")
    y_test = read.csv(paste0("./",data,"/",seed,"/test_y.txt"))
    
    xgboost_fit = xgboost(data.matrix(x_train), data.matrix(y_train), nrounds=30)
    xgboost_pred = predict(xgboost_fit, data.matrix(x_test))
    mae_xgboost = mean(abs(xgboost_pred-data.matrix(y_test)))
    mae_list_xgb[i,j] = mae_xgboost
    
    print(data)
    print(seed)
    print(mae_xgboost)
    
    fit_tree = rpart(data.matrix(y_train) ~., data=x_train)
    tree_pred = predict(fit_tree, x_test)
    mae_tree = mean(abs(tree_pred-data.matrix(y_test)))
    print(mae_tree)
    mae_list_tree[i,j] = mae_tree
  }
}

setwd("/Users/daisyding/Desktop/2020_Spring/MDI_GNN/uci/data_with_missing")
write.csv(mae_list_tree, "tree_0.3.txt")

xgb_mae_average = apply(mae_list_xgb,1,mean)
tree_mae_average = apply(mae_list_tree,1,mean)

write.csv(xgb_mae_average, "xgb_mae.txt")
write.csv(tree_mae_average, "tree_mae.txt")


#simulation
set.seed(3)
n = 100
p = 50
nv = 5 #true rank
n_sim = 30

x = matrix(rnorm(n*p),n,p)
b = matrix(c(5,3,2,rep(1, times=47)), ncol=1)
y = x %*% b

x_miss = x
n_entries = length(x_miss)

for (m in 1:(trunc(n_entries/5))){
  #cat(m,fill=T)
  i = sample(1:nrow(x_miss),size=1)
  j = sample(1:ncol(x_miss),size=1)
  x_miss[i,j] = NA
}

train_ind = sample(seq(100), 70)
x_train = x_miss[train_ind,]
y_train = y[train_ind]

x_test = x_miss[-train_ind,]
y_test = y[-train_ind]

xgboost_fit = xgboost(x_train, y_train, nrounds=5)
xgboost_pred = predict(xgboost_fit, x_test)
mean((xgboost_pred-y_test)^2)

fit_tree = rpart(y_train ~., data=as.data.frame(x_train))
tree_pred = predict(fit_tree, as.data.frame(x_test))
mean((tree_pred-y_test)^2)

#fit_forest = randomForest(y_train ~., data=as.data.frame(x_train))
#forest_pred = predict(fit_forest, as.data.frame(x_test))
#mean((forest_pred-y_test)^2)

