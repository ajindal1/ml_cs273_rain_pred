library(data.table)
library(xgboost)
library('sigr')
# This is to calculate Information Gain
library(CORElearn)

setwd('/Users/abhishekjindal/Desktop/UCI_courses/Fall_2017/CS273A/kaggleProj/mlTechniques')
train_x = data.table(read.table('../final_datasets/X_training_100K.txt', sep = ','))
train_y = data.table(read.table('../final_datasets/Y_training_100K.txt', sep = ','))
val_x = data.table(read.table('../final_datasets/X_validation_50K.txt', sep = ','))
val_y = data.table(read.table('../final_datasets/Y_validation_50K.txt', sep = ','))
test_x = data.table(read.table('../final_datasets/X_test_50K.txt', sep = ','))
test_y = data.table(read.table('../final_datasets/Y_test_50K.txt', sep = ','))


#train_x = train_x[, "V11" := NULL]
#val_x = val_x[, "V11" := NULL]
#train_x$target = train_y$V1
#val_x$target = val_y$V1

dval <- xgb.DMatrix(data=data.matrix(val_x),
                    label=data.matrix(val_y[,"V1",with=FALSE]),missing=NA)
dval <- xgb.DMatrix(data=data.matrix(train_x),
                    label=data.matrix(train_y[,"V1",with=FALSE]),missing=NA)

watchlist<-list(dval=dval)

clf <- xgb.train(params=list(  objective="binary:logistic", 
                               booster = "gbtree",
                               eta=0.15, 
                               max_depth=12, 
                               subsample=0.85,
                               colsample_bytree=1) ,
                 data = xgb.DMatrix(data=data.matrix(train_x),
                                    label=data.matrix(train_y[,"V1",with=FALSE]),missing=NA), 
                 nrounds = 60, 
                 verbose = 1,
                 print_every_n=5,
                 early_stopping_rounds    = 10,
                 watchlist           = watchlist,
                 maximize            = FALSE,
                 eval_metric='logloss'
)
pred_train<-predict(clf,xgb.DMatrix(data.matrix(train_x),missing=NA))
calcAUC(modelPredictions = pred_train, yValues = as.logical(train_y$V1))

pred_val<-predict(clf,xgb.DMatrix(data.matrix(val_x),missing=NA))
calcAUC(modelPredictions = pred_val, yValues = as.logical(val_y$V1))

#pred_val<-predict(clf,xgb.DMatrix(data.matrix(test_x),missing=NA))
#calcAUC(modelPredictions = pred_val, yValues = as.logical(test_y$V1))
#head(pred_val)
#head(test_y$V1)
