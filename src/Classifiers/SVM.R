library(e1071)
library(ROSE)
library(imbalance)
library(tidyr)
library(kernlab)

setwd("C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\Classifiers")
getwd()
load('covid_data_benchmark')
dataset <- covid_data
summary(dataset)





#Factoring the Target variable of (0,1)
df$V59 = factor(dataset$V59, levels = c(0, 1))
print(is.factor(dataset$V59))



#Over-sampling
df.blsmote <- oversample(dataset, ratio = 0.8, method = 'BLSMOTE', filtering = FALSE, classAttr = "V59", wrapper = c("KNN", "C5.0"))
#imbalanceRatio(df, classAttr = "V59")
# df.adasyn <- oversample(df, method = "ADASYN", classAttr = "V59")
# #imbalanceRatio(df, classAttr = "V59")
# df.dbsmote <- oversample(df, ratio = 0.8, method = "DBSMOTE", filtering = FALSE, classAttr = "V59", wrapper = c("KNN", "C5.0"))
# #imbalanceRatio(df, classAttr = "V59")
# df.random <- ovun.sample(V59~., df, method = "over")$data
# table(df.random$V59)


# kpca(df.blsmote, kernel = "rbfdot", kpar = list(sigma = 0.1),
#      features = 58, th = 1e-4, na.action = na.omit)
# 
# # S4 method for kernelMatrix
# k.pca <- kpca(df.blsmote, features = 2, th = 1e-4)
# pcv(k.pca)
# plot(rotated(k.pca),col=as.integer(df[-df.blsmote,5]),
#      xlab="1st Principal Component",ylab="2nd Principal Component")
# summary(k.pca)
# 
# # S4 method for list
# kpca(x, kernel = "stringdot", kpar = list(length = 4, lambda = 0.5),
#      features = 58, th = 1e-4, na.action = na.omit, ...)



#train-test split
index <- 1:nrow(df.blsmote)
testindex <- sample(index, trunc(length(index)/3))
testset <- df.blsmote[testindex,]
trainset <- df.blsmote[-testindex,]



set.seed(825)
#fitting the Support Vector Machine to the Training set
svm_fit_ovs <- svm(V59~., kernel = 'radial',
               data = trainset, scale=TRUE, cachesize = 200,
               probability = TRUE, gamma = 0.1, cost = 1)
summary(svm_fit_ovs)



#hyper-parameter tuning
obj <- tune.svm(V59~., data = trainset, scaled =TRUE,
          gamma = 10^c(-1:10), cost = (1:100), 
          tune.control(nrepeat = 5, sampling = "cross", cross = 10), probability = TRUE)
summary(obj)
obj
svm_fit_ovs <- obj$best.model
print(svm_fit_ovs)



#Predicting the test set results
#svm.pred <- predict(svm_fit, testset[,-59])
svm.pred.ovs <- predict(svm_fit_ovs, testset[,-59], probability = TRUE)



#plot(svm.pred)
plot(svm.pred.ovs)
summary(svm.pred.ovs)



#accuracy.meas(response = testset$V59, predicted = svm.pred)
accuracy.meas(response = testset$V59, predicted = svm.pred.ovs)

#roc.curve(testset$V59, svm.pred, plotit = T)
roc.curve(testset$V59, svm.pred.ovs, plotit = T)

#confusionMatrix(svm.pred, testset$V59, positive = '1')
caret::confusionMatrix(svm.pred.ovs, testset$V59, positive = '1')


# library(caret)
# set.seed(12345)
# # Setup for cross validation
# ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
#                      repeats=5,         # do 5 repetitions of cv
#                      summaryFunction=twoClassSummary,   # Use AUC to pick the best model
#                      classProbs=TRUE)
# 
# 
# #Train and Tune the SVM
# svm.tune <- train(x=xdata,
#                   y= ydata,
#                   method = "svmRadial",   # Radial kernel
#                   tuneLength = 5,                   # 5 values of the cost function
#                   preProc = c("center","scale"),  # Center and scale data
#                   metric="ROC",
#                   trControl=ctrl)
# 
# svm.tune
# plot(svm.tune)


#save(svm_fit, file = "SVM_raw")
save(svm_fit_ovs, file = "SVM_ovs_benchmark")
