library(EMCluster)
library(data.table)

setwd('/Users/abhishekjindal/Desktop/UCI_courses/Fall_2017/CS273A/kaggleProj/mlTechniques')
train_x = data.table(read.table('../final_datasets/X_training_100K.txt', sep = ','))
train_y = data.table(read.table('../final_datasets/Y_training_100K.txt', sep = ','))
val_x = data.table(read.table('../final_datasets/X_validation_50K.txt', sep = ','))
val_y = data.table(read.table('../final_datasets/Y_validation_50K.txt', sep = ','))


sample_data = train_x[0:30000]
sample_y = train_y[0:nrow(sample_data)]
nrow(sample_data)
numClasses = 5
ret <- init.EM(sample_data, nclass = numClasses)
ret.new <- assign.class(sample_data, ret, return.all = FALSE)
str(ret.new)
for(i in seq(numClasses)){
  print(mean(sample_y[ret.new["class"]$class == i]$V1))
}

ret.new <- assign.class(train_x, ret, return.all = FALSE)
for(i in seq(numClasses)){
  print(mean(train_y[ret.new["class"]$class == i]$V1))
}

ret.new <- assign.class(val_x, ret, return.all = FALSE)
for(i in seq(numClasses)){
  print(mean(val_y[ret.new["class"]$class == i]$V1))
}

for(i in seq(numClasses)){
  print(length(val_y[ret.new["class"]$class == i]$V1))
}
