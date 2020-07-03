library(lime)    # for explaining models
library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting
library(wvtool)
library(imager)
library(abind)
library(caret)

setwd('C:/Users/jalaj/Documents/DataSci with R/Project Files')
par(mfrow=c(2,2))

#load("knn_model")
load("svm_oversampled")


extract_features <- function(fpath){
  img = load.image(fpath)
  plot(img)
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


wraped <- function(img_path){
  y = extract_features(img_path)
  inf = matrix(data = c(y), ncol = length(y), nrow = 1)
  z = as.data.frame(inf)
  return(z)
}


do_prediction <- function(img_path)
{

  #img_path <- "D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-positive\\1B734A89-A1BF-49A8-A1D3-66FAFA4FAC5D.jpeg"
  
  z = wraped(img_path)
  
  
  test_pred <- predict(svm_fit_ovs, newdata = z, type = c(type))
  return(test_pred)
}

#do_prediction("D:\\dke\\2ND SEM\\IRTEX\\R&D\\ibm_explain\\lime\\3.JPG")

# 
# 
# 
# predict_model.ensemble <- function(x, newdata, type, ...) {
#   print("here inside model ensemble")
#   print(newdata)
#   res <- predict(x, newdata = newdata, type = c(type), ...)
#   print(res)
#   res = c(res)
#   cc = c()
#   # for (i in res)
#   # {
#   #   
#   #   if(i == 0)
#   #   {
#   #     cc = c(cc, "negative")
#   #   }
#   #   else
#   #   {
#   #     cc = c(cc, "positive")
#   #   }
#   #   
#   # }
#   print(cc)
#   print(type)
#   switch(
#     type,
#     raw = data.frame(Response = cc, stringsAsFactors = FALSE),
#     prob = as.data.frame(res, check.names = FALSE)
#   )
# }
# 
# model_type.ensemble <- function(x, ...) 'classification'
# 
# 
# 
# 
# #x =print(do_prediction("D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-negative\\1.jpg"))
# 
# i = 0
# 
# img_path <- "D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-negative\\1.jpg"
# 
# #z = wraped(img_path)
# 
# img_preprocess <- function(x) {
#   arrays <- lapply(x, function(path) {
#     print(paste("inside preoprocess", path))
#     return(wraped(path))
#   })
#   do.call(abind, c(arrays, list(along = 1)))
# }
# 
# img_path = "D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-negative\\1.jpg"
# #img_path = "D:/dke/2ND SEM/R Project/resource/covid-dataset/covid-dataset/covid-positive/1B734A89-A1BF-49A8-A1D3-66FAFA4FAC5D.jpeg"
# img_path2 = "D:/dke/2ND SEM/R Project/resource/covid-dataset/covid-dataset/covid-positive/6b3bdbc31f65230b8cdcc3cef5f8ba8a-40ac-0.jpg"
# 
# 
# ensemble_obj = knn_fit
# class(ensemble_obj) = c("ensemble", class(knn_fit))
# 
# 
# explainer2 <- lime(c(img_path, img_path2), as_classifier(knn_fit, NULL), img_preprocess)
# 
# explanation2 <- explain(c(img_path, img_path2), explainer2, n_labels = 2,
#                         n_features = 2, n_permutations = 20, background = "white")
# 
# 
# plot_image_explanation(explanation2, display = 'block', threshold = 5e-07)
# 
# 
# 
# 
# 
