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


set.seed(100)
clf <- xgb.train(params=list(  objective="binary:logistic", 
                               booster = "gbtree",
                               eta=0.15, 
                               max_depth=12, 
                               subsample=0.85,
                               colsample_bytree=1) ,
                 data = xgb.DMatrix(data=data.matrix(train_x),
                                    label=data.matrix(train_y[,"V1",with=FALSE]),missing=NA), 
                 nrounds = 70, 
                 verbose = 1,
                 print_every_n=5,
#                 early_stopping_rounds    = 10,
#                 watchlist           = watchlist,
                 maximize            = FALSE,
                 eval_metric='logloss'
)
pred_train<-predict(clf,xgb.DMatrix(data.matrix(train_x),missing=NA))
calcAUC(modelPredictions = pred_train, yValues = as.logical(train_y$V1))

pred_val<-predict(clf,xgb.DMatrix(data.matrix(val_x),missing=NA))
calcAUC(modelPredictions = pred_val, yValues = as.logical(val_y$V1))

pred_test<-predict(clf,xgb.DMatrix(data.matrix(test_x),missing=NA))
calcAUC(modelPredictions = pred_test, yValues = as.logical(test_y$V1))
#head(pred_val)
#head(test_y$V1)

#Making final data set

train_x_total <- rbind(train_x, val_x, test_x)
train_y_total <- rbind(train_y, val_y, test_y)

set.seed(100)
clf <- xgb.train(params=list(  objective="binary:logistic", 
                               booster = "gbtree",
                               eta=0.15, 
                               max_depth=15, 
                               subsample=0.65,
                               colsample_bytree=1) ,
                 data = xgb.DMatrix(data=data.matrix(train_x_total),
                                    label=data.matrix(train_y_total[,"V1",with=FALSE]),missing=NA), 
                 nrounds = 80, 
                 verbose = 1,
                 print_every_n=5,
#                 early_stopping_rounds    = 10,
#                 watchlist           = watchlist,
                 maximize            = FALSE,
                 eval_metric='logloss'
)


pred_train_total<-predict(clf,xgb.DMatrix(data.matrix(train_x_total),missing=NA))
calcAUC(modelPredictions = pred_train_total, yValues = as.logical(train_y_total$V1))

submission_x = data.table(read.table('../final_datasets/X_submission_200K.txt', sep = ','))
pred_submission<-predict(clf,xgb.DMatrix(data.matrix(submission_x),missing=NA))
head(pred_submission)
length(pred_submission)

submission_table <- data.frame(seq(0, length(pred_submission)-1), pred_submission)
names(submission_table) <- c('Id', 'Prob1')
write.csv(submission_table, "Y_submission_xgboost_v2.csv", row.names = F)

# This got us a score of 0.78983 on leaderboard!
# This got us a score of 0.78346 on leaderboard!

temp <- read.table('Y_submit_basic_ensemble_DT_NN.txt', sep=',', header = TRUE)

nrow(temp[temp$Prob1 > 0.9,])
nrow(submission_table[submission_table$Prob1 > 0.9,])

final_table <- copy(submission_table)
final_table[(submission_table$Prob1 > 0.9 & temp$Prob1 > 0.9),]$Prob1 <- max(temp[(submission_table$Prob1 > 0.9 & temp$Prob1 > 0.9),]$Prob1, final_table[(submission_table$Prob1 > 0.9 & temp$Prob1 > 0.9),]$Prob1)
final_table[(submission_table$Prob1 < 0.1 & temp$Prob1 < 0.1),]$Prob1 <- min(temp[(submission_table$Prob1 < 0.1 & temp$Prob1 < 0.1),]$Prob1, final_table[(submission_table$Prob1 < 0.1 & temp$Prob1 < 0.1),]$Prob1)

mean((final_table$Prob1 - submission_table$Prob1)^2)
temp_final <- copy(final_table)
final_table[(submission_table$Prob1 > 0.1 & temp$Prob1 > 0.1 & submission_table$Prob1 < 0.9 & temp$Prob1 < 0.9),]$Prob1 <- 
  0.8*final_table[(submission_table$Prob1 > 0.1 & temp$Prob1 > 0.1 & submission_table$Prob1 < 0.9 & temp$Prob1 < 0.9),]$Prob1 +
  0.2*temp[(submission_table$Prob1 > 0.1 & temp$Prob1 > 0.1 & submission_table$Prob1 < 0.9 & temp$Prob1 < 0.9),]$Prob1


write.csv(final_table, "Y_submission_xgboost_v2_ensemble.csv", row.names = F)
