library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting
library(wvtool)
library(imager)
library(caret)
library(crul)
library(jsonlite)
library(e1071)

setwd('C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\app_with_mask\\explain_server')
par(mfrow=c(3,2))

#load("knn_model_masked")
load("svm_ovs_masked")

extract_features <- function(fpath){
  plot(load.image(fpath))
  url_path_mask = "http://127.0.0.1:8000"
  x <- HttpClient$new(url = url_path_mask)
  y = x$get(path = "mask", query = list("fileName"=fpath, 
                                        "workdir"="C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\app_with_mask\\explain_server"))
  z = fromJSON(y$parse())
  #plot(load.image(z$name))
  url_path_lbp = "http://127.0.0.1:4000"
  x <- HttpClient$new(url = url_path_lbp)
  y = x$get(path = "lbp", query = list("fileName"=fpath, "maskName"=z$fname))
  z = fromJSON(y$parse())
  return (z$lbp)
  
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
  
  
  #test_pred <- predict(knn_fit, newdata = z, type = c('prob'))
  test_pred_svm <- predict(svm_fit_ovs, newdata = z, probability = TRUE)
  test_pred_svm <- attr(test_pred_svm, "probabilities")
  test_pred_svm <- data.frame(test_pred_svm)
  
  #averaging
  
  return(test_pred_svm)
}

wraped_mask <- function(img_path, mask_path){
  
  url_path_lbp = "http://127.0.0.1:4000"
  x <- HttpClient$new(url = url_path_lbp)
  y = x$get(path = "lbp", query = list("fileName"=img_path, "maskName"=mask_path))
  z = fromJSON(y$parse())
  y = z$lbp
  inf = matrix(data = c(y), ncol = length(y), nrow = 1)
  z = as.data.frame(inf)
  
  return(z)
}

do_prediction_with_mask <- function(img_path, mask_path, original)
{
  plot(load.image(img_path))
  plot(load.image(mask_path))
  plot(load.image(original))
  print("hello from r")
  #img_path <- "D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-positive\\1B734A89-A1BF-49A8-A1D3-66FAFA4FAC5D.jpeg"
  
  z = wraped_mask(img_path, mask_path)
  
  
  #test_pred <- predict(knn_fit, newdata = z, type = c('prob'))
  test_pred_svm <- predict(svm_fit_ovs, newdata = z, probability = TRUE)
  test_pred_svm <- attr(test_pred_svm, "probabilities")
  test_pred_svm <- data.frame(test_pred_svm)
  #averaging
  
  return(test_pred_svm)
}

controller <- function(img, wd, url_path)
{
  print("doing prediction")
  p = do_prediction(img_path = img)
  print("done prediction")
  print("doing explanation")
  x <- HttpClient$new(url = url_path)
  y = x$get(path = "explain", query = list("image"=img, "exp"=wd))
  z = fromJSON(y$parse())
  print("done expanation")
  print(p)
  plot(load.image(z$name))
}
