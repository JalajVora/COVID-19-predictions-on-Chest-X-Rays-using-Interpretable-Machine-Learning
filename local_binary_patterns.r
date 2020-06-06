library(wvtool)
library(imager)

par(mfrow=c(2,2))



#this is local bindary pattern histogram

extract_features <- function(fpath){
  im = load.image(fpath)
  #plot(im)
  img <- grayscale(im)
  #plot(img)
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


files <- list.files("C:\\Users\\jalaj\\Documents\\DataSci with R\\covid-dataset", recursive = TRUE, full.names = TRUE, pattern = "*jpg")

covid_data = NULL

for (file in files)
{
  hist <- extract_features(file)
  
  x= 0
  if(grepl("negative", file))
    x = 1
  
  #dat <- list("filename"=file, "feature"=hist, "target"=1)
  hist <- append(hist,  x)

  inf = matrix(data = c(hist), ncol = length(hist), nrow = 1)
  covid_data <- rbind(covid_data, inf)
  

  print(file)
}

covid_data = as.data.frame(covid_data)
print(covid_data)
library(e1071)


#Plotting Dataset to see how it looks like
#plot(iris)
# s<- sample(150,100)
# col<- c("Petal.Length", "Petal.Width", "Species")
# iris_train <- iris[s,col]
# iris_test <- iris[-s,col]
# 
# #Building SVM-Linear Model
# svmfit <- svm(Species~., data=iris_train, kernel="linear", cost=0.1, scale=FALSE)
# print(svmfit)
# plot(svmfit, iris_train[,col])
# 
# 
# #Cross-Validation Using tune() function:
# tuned <- tune(svm, Species~., data=iris_train, kernel="linear", ranges = list(cost=c(0.001, 0.01, 0.1, 1, 10, 100)))
# summary(tuned)
# 
# 
# #Predicting using Test set
# p <- predict(svmfit, iris_test[,col], type="class")
# plot(p)
# table(p, iris_test[,3])
# mean(p == iris_test[,3])