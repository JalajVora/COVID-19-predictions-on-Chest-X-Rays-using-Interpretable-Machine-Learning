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


files <- list.files("D:/dke/2ND SEM/R Project/resource/covid-dataset/covid-dataset", recursive = TRUE, full.names = TRUE, pattern = "*jpg")

covid_data = NULL

for (file in files)
{
  hist <- extract_features(file)
  
  x= 0
  if(grepl("negetive", file))
    x = 1
  
  #dat <- list("filename"=file, "feature"=hist, "target"=1)
  hist <- append(hist,  x)

  inf = matrix(data = c(hist), ncol = length(hist), nrow = 1)
  covid_data <- rbind(covid_data, inf)
  

  print(file)
}

covid_data = as.data.frame(covid_data)




















