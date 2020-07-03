library(wvtool)
library(imager)
library(class)
library(magick)
library(crul)
library(jsonlite)

par(mfrow=c(2,2))

setwd("C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\lung-segmentation-2d-master")


#this is local bindary pattern histogram

extract_features <- function(fpath){
  # img = load.image(fpath)
  # 
  # if(length(channels(img))>3){
  #   img = image_read(fpath)
  #   img = image_convert(img, "jpeg")
  #   image_write(img, "try.jpg")
  #   img = load.image("try.jpg")
  #   file.remove("try.jpg")
  # }
  # 
  # #plot(im)
  # if(length(channels(img))!=1)
  # {
  #   img <- grayscale(img)
  # }
  url_path_mask = "http://127.0.0.1:8000"
  x <- HttpClient$new(url = url_path_mask)
  y = x$get(path = "mask", query = list("fileName"=fpath, "workdir"="C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\covid-dataset\\covid_data_masked"))
  z = fromJSON(y$parse())
  
  url_path_lbp = "http://127.0.0.1:4000"
  x <- HttpClient$new(url = url_path_lbp)
  y = x$get(path = "lbp", query = list("fileName"=fpath, "maskName"=z$fname))
  z = fromJSON(y$parse())
  
  return(data.frame(t((z$lbp))))
  
  #plot(img)
  #call the sever for the image mask- filename and working dir inp, mask file path as output
  #imgm <- data.matrix(img)
  #lbpd <- lbp(imgm, 2)
  #call the server for lbp with mask - filename and mask file path as input.. a array of numbers as return
  #data.frame()
  #h_lbp = hist(lbpd$lbp.u2, breaks = 59, plot=FALSE)
  ##################
  #implement other descriptors here
  
  #do early fusion here, that is attach multiple other descriptors to form a single descriptor
  
  #this is makeshift
  #return(h_lbp$counts)
  
}
#hist = extract_features(fpath)
#print(hist)


files <- list.files("C:\\Users\\jalaj\\Documents\\DataSci with R\\Project Files\\covid-dataset\\covid-positive", recursive = TRUE, full.names = TRUE, pattern = "*.*")

covid_data = NULL

for (file in files)
{
  
    gc()
    
    print(file)
    
    hist <- extract_features(file)
    
    x= 1
    if(grepl("negative", file))
      x = 0
    
    
    hist <- append(hist,  x)
    
    inf = matrix(data = c(hist), ncol = length(hist), nrow = 1)
    covid_data <- rbind(covid_data, inf)
    
}

covid_data = as.data.frame(covid_data)
save(covid_data, file= "covid_data_positive")