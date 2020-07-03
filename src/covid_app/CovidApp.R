library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting
library(wvtool)
library(imager)
library(caret)
library(crul)
library(jsonlite)
library(e1071)

setwd("C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\app_with_mask\\covid_app")

par(mfrow=c(2,2))

#load("knn_model_masked")
load("SVM_ovs_masked")


extract_features <- function(fpath){
  
  url_path_mask = "http://127.0.0.1:8000"
  x <- HttpClient$new(url = url_path_mask)
  y = x$get(path = "mask", query = list("fileName"=fpath, 
                                        "workdir"="C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\covid_data_masked"))
  z = fromJSON(y$parse())
  
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
  
  
  #test_pred_knn <- predict(knn_fit, newdata = z, type = c('prob'))
  test_pred_svm <- predict(svm_fit_ovs, newdata = z, probability = TRUE)
  test_pred_svm <- attr(test_pred_svm, "probabilities")
  test_pred_svm <- data.frame(test_pred_svm)
  print(test_pred_svm)
  #print(test_pred_knn)
  #averaging
  
  #for (i in 1:3)
  {
    #zero_prob <- mean(c(test_pred_knn[i][1], test_pred_svm[i][1]))
    #ones_prob <- mean(c(test_pred_knn[i][2], test_pred_svm[i][2]))
  }
  #test_pred <- data.frame(rbind(zero_prob, ones_prob))
  #return(test_pred)
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

#this the calling application calls
files <- list.files("C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\covid-dataset\\covid-negative", recursive = TRUE, full.names = TRUE, pattern = "*.*")
for (file in files)
{
  #print(do_prediction(file))
  controller(img = file, wd = "C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\covid_data_masked", url_path = "http://127.0.0.1:5000")
}
#controller(img = "D:/dke/2ND SEM/IRTEX/R&D/ibm_explain/lime/4.jpg", wd = "D:/dke/2ND SEM/IRTEX/R&D/ibm_explain/lime/wd", url_path = "http://127.0.0.1:5000")

