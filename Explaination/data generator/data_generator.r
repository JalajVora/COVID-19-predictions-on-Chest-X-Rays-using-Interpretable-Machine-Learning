library(wvtool)
library(imager)
library(class)
library(magick)

par(mfrow=c(2,2))



#this is local bindary pattern histogram

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
#hist = extract_features(fpath)
#print(hist)


files <- list.files("D:/dke/2ND SEM/R Project/resource/covid-dataset/covid-dataset/covid-positive", recursive = TRUE, full.names = TRUE, pattern = "*jpg")

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