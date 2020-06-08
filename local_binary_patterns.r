library(wvtool)
library(imager)

par(mfrow=c(2,2))



#this is local bindary pattern histogram

extract_features <- function(fpath){
  im = load.image(fpath)
  #plot(im)
  img <- grayscale(im)
  #plot(img)
  #rgb2gray(img, c(1,1,1))
  imgm <- data.matrix(img)
  lbpd <- lbp(imgm, 2)
  h_lbp = hist(lbpd$lbp.u2, breaks = 59)
  ##################
  #implement other descriptors here
  
  #do early fusion here, that is attach multiple other descriptors to form a single descriptor
  
  #this is makeshift
  return(h_lbp$counts)
  
}

#hist = extract_features(fpath)
#print(hist)


files <- list.files("C:\\Users\\jalaj\\Documents\\DataSci with R\\covid-data", recursive = TRUE, full.names = TRUE, pattern = "*jpg")

covid_data = NULL

for (file in files)
{
  hist <- extract_features(file)
  
  x= 0
  if(grepl("covid-positive", file))
    x = 1
  
  #dat <- list("filename"=file, "feature"=hist, "target"=1)
  hist <- append(hist,  x)

  inf = matrix(data = c(hist), ncol = length(hist), nrow = 1)
  covid_data <- rbind(covid_data, inf)
  

  print(file)
}

covid_data = as.data.frame(covid_data)
dim(covid_data)
str(covid_data)
summary(covid_data)

library(e1071)
library(caret)
library(mlr3)


set.seed(3033)
intrain <- createDataPartition(y=covid_data$V59, p = 0.7, list = FALSE)
training <- covid_data[intrain,]
testing <- covid_data[-intrain,]

dim(training)
dim(testing)

anyNA(covid_data)

summary(covid_data)

training [["V59"]] = factor(training[["V59"]])


#Building SVM-Linear Model
svmfit <- svm(covid_data$V59~., data=training, cost=0.1, scale=FALSE)
print(svmfit)
summary(svmfit)
plot(svmfit, train[,col])


#Cross-Validation Using tune() function:
tuned <- tune(svm, V59~., data=train, kernel="linear", ranges = list(cost=c(0.001, 0.01, 0.1, 1, 10, 100)))
summary(tuned)


#Predicting using Test set
p <- predict(svmfit, test[,col], type="class")
plot(p)
table(p, test[,3])
mean(p == test[,3])