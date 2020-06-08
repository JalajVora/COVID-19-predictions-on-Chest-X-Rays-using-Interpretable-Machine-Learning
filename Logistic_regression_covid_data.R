dataset <- covid_data

#getwd()
#splitting the data into training set and test set
library(caTools)
set.seed(123)


split <- sample.split(dataset, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

#scaling

training_set[, -(length(training_set))] <- scale(training_set[, -(length(training_set))])

test_set[, -(length(test_set))] <- scale(test_set[, -(length(test_set))])


#fitting the Logistic Regression to the Training set
classifier <- glm(V59 ~.,
                  family = binomial,
                  data = training_set)

#Predicting the test set results
prob_pred <- predict(classifier, type = 'response', newdata = test_set[, -(length(test_set))])
y_pred <- ifelse(prob_pred >0.5, 1, 0)

#making the Confusion Matrix
cm <- table(test_set$V59, y_pred)
cm