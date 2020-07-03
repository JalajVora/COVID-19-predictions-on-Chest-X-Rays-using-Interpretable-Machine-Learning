setwd("C:/Users/jalaj/Documents/DataSci with R/Project Files/Classifiers")
load('covid_data_masked')
data <- covid_data_masked
#Setting V59 as factor variable as that is the target
data$V59 <- as.factor(data$V59)
summary(data)

#Split Dataset to train and test
set.seed(123) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(data), size = floor(.75*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]

#Visualizing distribution of each class 
barplot(prop.table(table(train$V59)),
        col = rainbow(2),
        ylim = c(0,1),
        main = "Class Distribution")
barplot(prop.table(table(test$V59)),
        col = rainbow(2),
        ylim = c(0,1),
        main = "Class Distribution")

#Training Models
library(randomForest)
library(rpart)
library(rpart.plot)
#Random Forest
rftrain <- randomForest(V59~.,data = train)
#Decision Tree
dttrain <- rpart(V59~.,data = train, method = "class")
dttrain_criteria <- rpart(V59~.,data = train, method = "class", minsplit = 20, minbucket = 10)
rpart.plot(dttrain)
rpart.plot(dttrain_criteria)

save(dttrain, file = "DT_model")
save(rftrain, file = "RF_model")

#Evaluation with test
library(caret)
confusionMatrix(predict(rftrain, test), test$V59, positive = '1')
confusionMatrix(predict(object=dttrain,test,type="class"), test$V59, positive = '1')
confusionMatrix(predict(object=dttrain_criteria,test,type="class"), test$V59, positive = '1')

#Under Sampling
library(ROSE)
under_train = ovun.sample(V59~., data = train, method = "under", N = 294)$data
table(under_train$V59)
under_test = ovun.sample(V59~., data = test, method = "under", N = 100)$data
table(under_test$V59)

barplot(prop.table(table(under_train$V59)),
        col = rainbow(2),
        ylim = c(0,1),
        main = "Class Distribution")
barplot(prop.table(table(under_test$V59)),
        col = rainbow(2),
        ylim = c(0,1),
        main = "Class Distribution")

library(randomForest)
rftrain_under <- randomForest(V59~.,data = under_train)
dttrain_under <- rpart(V59~.,data = under_train, method = "class")
rpart.plot(dttrain_under)

#Evaluation with test
library(caret)
confusionMatrix(predict(rftrain_under, under_test), under_test$V59, positive = '1')
confusionMatrix(predict(object=dttrain_under,under_test,type="class"), under_test$V59, positive = '1')

#Over Sampling
library(ROSE)
over_train = ovun.sample(V59~., data = train, method = "over", N = 1794)$data
table(over_train$V59)
over_test = ovun.sample(V59~., data = test, method = "over", N = 600)$data
table(over_test$V59)

barplot(prop.table(table(over_train$V59)),
        col = rainbow(2),
        ylim = c(0,1),
        main = "Class Distribution")
barplot(prop.table(table(over_test$V59)),
        col = rainbow(2),
        ylim = c(0,1),
        main = "Class Distribution")

library(randomForest)
rftrain_over <- randomForest(V59~.,data = over_train)
dttrain_over <- rpart(V59~.,data = over_train, method = "class")
rpart.plot(dttrain_over)

save(dttrain_over, file = "DT_model_over")
save(rftrain_over, file = "RF_model_over")

#Evaluation with test
library(caret)
confusionMatrix(predict(rftrain_over, over_test), over_test$V59, positive = '1')
confusionMatrix(predict(object=dttrain_over,over_test,type="class"), over_test$V59, positive = '1')

