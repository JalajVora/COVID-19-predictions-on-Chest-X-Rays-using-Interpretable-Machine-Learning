#Loading Packages and Libraries
#install.packages("caret")
#install.packages("caretEnsemble")
#library(caret)
library(e1071)


#Plotting Dataset to see how it looks like
plot(iris)
s<- sample(150,100)
col<- c("Petal.Length", "Petal.Width", "Species")
iris_train <- iris[s,col]
iris_test <- iris[-s,col]

#Building SVM-Linear Model
svmfit <- svm(Species~., data=iris_train, kernel="linear", cost=0.1, scale=FALSE)
print(svmfit)
plot(svmfit, iris_train[,col])


#Cross-Validation Using tune() function:
tuned <- tune(svm, Species~., data=iris_train, kernel="linear", ranges = list(cost=c(0.001, 0.01, 0.1, 1, 10, 100)))
summary(tuned)


#Predicting using Test set
p <- predict(svmfit, iris_test[,col], type="class")
plot(p)
table(p, iris_test[,3])
mean(p == iris_test[,3])