library(caret)
library(ggplot2)
library(wvtool)
library(imager)

#glass.df <-read.csv("D:/dke/2ND SEM/R Project/resource/glass.csv")
load("covid_data")
glass.df = covid_data

dim(glass.df)
names(glass.df)
head(glass.df)
tail(glass.df)

summary(glass.df)
str(glass.df)

# all the independent variables are numbers
# we will convert the type variable which is the response variable as factor

glass.df$V59<- as.factor(glass.df$V59) # 7 labels

# training and test data 70:30
set.seed(123)
ind = sample(2, nrow(glass.df), replace = TRUE, prob=c(0.7,0.3))
train.df = glass.df[ind == 1,]
test.df = glass.df[ind == 2,]
dim(train.df)
dim(test.df)

table(train.df$V59)

table(test.df$V59)

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10,
  savePredictions = TRUE
)

## Logistic regression
knn_fit<-train(V59 ~.,data=train.df,method="glm",family=binomial(),trControl=fitControl, type = "response")


#trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#knn_fit <- train(V59 ~., data = train.df, method = "knn",trControl=trctrl,preProcess = c("center", "scale"),tuneLength = 15)

knn_fit

plot(knn_fit)

test_pred <- predict(knn_fit, newdata = test.df)
test_pred


confusionMatrix(test_pred, test.df$V59 )


mean(test_pred == test.df$V59)


extract_features <- function(fpath){
  img = load.image(fpath)
  
  if(length(channels(img))>3){
    img = image_read(fpath)
    img = image_convert(img, "jpeg")
    image_write(img, "try.jpg")
    img = load.image("try.jpg")
    file.remove("try.jpg")
  }
  
  #plot(im)
  if(length(channels(img))!=1)
  {
    img <- grayscale(img)
  }
  #plot(img)
  imgm <- data.matrix(img)
  lbpd <- lbp(imgm, 2)
  h_lbp = hist(lbpd$lbp.u2, breaks = 59, plot=FALSE)
  ##################
  #implement other descriptors here
  
  #do early fusion here, that is attach multiple other descriptors to form a single descriptor
  
  #this is makeshift
  return(h_lbp$counts)
  
}
img_path <- "D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-positive\\1B734A89-A1BF-49A8-A1D3-66FAFA4FAC5D.jpeg"
img_path = "D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-negative\\4.jpg"
#img_path = "D:/dke/2ND SEM/IRTEX/R&D/ibm_explain/lime/3.jpg"
y = extract_features(img_path)
inf = matrix(data = c(y), ncol = length(y), nrow = 1)
z = as.data.frame(inf)



test_pred <- predict(knn_fit, newdata = z, type = c('prob'))
test_pred

save(knn_fit, file = "knn_model")

#confusionMatrix(test_pred, test.df$V59 )

#mean(test_pred == test.df$V59)
