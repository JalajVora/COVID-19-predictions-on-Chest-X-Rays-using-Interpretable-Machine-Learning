library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting
library(wvtool)
library(imager)
library(abind)
library(caret)


par(mfrow=c(2,2))

load("knn_model")

  print(knn_fit)

extract_features <- function(fpath){
  
  img = load.image(fpath)
  plot(img)
  if(length(channels(img))>3){
    print("inside the first error")
    img = image_read(fpath)
    img = image_convert(img, "jpeg")
    image_write(img, "try.jpg")
    img = load.image("try.jpg")
    file.remove("try.jpg")
  }
  
  #plot(im)
  if(length(channels(img))!=1)
  {
    print("inside the second error")
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
  #print(h_lbp$counts)
  return(h_lbp$counts)
  
}


wraped <- function(img_path){
  y = extract_features(img_path)
  inf = matrix(data = c(y), ncol = length(y), nrow = 1)
  z = as.data.frame(inf)
  #print(z)
  return(z)
}


do_prediction <- function(img_path)
{
  
  #img_path <- "D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-positive\\1B734A89-A1BF-49A8-A1D3-66FAFA4FAC5D.jpeg"
  
  z = wraped(img_path)
  
  test_pred <- predict(knn_fit, newdata = z, type = c('prob'))
  print(test_pred)
  return(test_pred)
}


