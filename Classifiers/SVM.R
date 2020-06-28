#install.packages('e1071')
library(e1071)
#install.packages('ROSE')
library(ROSE)
#install.packages('caret')
library(caret)
#install.packages('tidyr')
library(tidyr)



setwd("C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\Classifiers")
getwd()
load('data_mask')
dataset <- covid_data
summary(covid_data)
dataset.summary()
dataset %>% drop_na()
#print(dataset$V59)

#Factoring the Target variable of (0,1)
dataset$V59 = factor(dataset$V59, levels = c(0, 1))
print(is.factor(dataset$V59))



index <- 1:nrow(dataset)
testindex <- sample(index, trunc(length(index)/3))
testset <- dataset[testindex,]
trainset <- dataset[-testindex,]


#over sampling
train_ov_sampled <- ovun.sample(V59 ~ ., trainset, method = "over")$data
table(train_ov_sampled$V59)

test_ov_sampled <- ovun.sample(V59 ~ ., testset, method = "over")$data
table(test_ov_sampled$V59)


set.seed(825)




#fitting the Support Vector Machine to the Training set
svm_fit <- svm(V59~., kernel = 'radial',
           data = trainset, scale=TRUE, cachesize = 200,
           probability = TRUE)

svm_fit_ovs <- svm(V59~., kernel = 'radial',
               data = train_ov_sampled, scale=TRUE, cachesize = 200,
               probability = TRUE)

#Predicting the test set results
svm.pred <- predict(svm_fit, testset[,-59])
svm.pred.ovs <- predict(svm_fit, test_ov_sampled[,-59])

plot(svm.pred)
plot(svm.pred.ovs)




accuracy.meas(response = testset$V59, predicted = svm.pred)
accuracy.meas(response = test_ov_sampled$V59, predicted = svm.pred.ovs)
roc.curve(testset$V59, svm.pred, plotit = T)
roc.curve(test_ov_sampled$V59, svm.pred.ovs, plotit = T)


confusionMatrix(svm.pred, testset$V59, positive = '1')
confusionMatrix(svm.pred.ovs, test_ov_sampled$V59, positive = '1')




save(svm_fit, file = "SVM_raw")
save(svm_fit_ovs, file = "SVM_oversampled")
