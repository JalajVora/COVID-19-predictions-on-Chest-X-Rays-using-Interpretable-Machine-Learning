import cv2
from PIL import Image

img_pil = Image.open('C:/Users/jalaj/Documents/DataSci with R/Project Files/covid-dataset/covid-positive/1109.jpg')
print('Pillow: ', img_pil.mode, img_pil.size)

img = cv2.imread('C:/Users/jalaj/Documents/DataSci with R/Project Files/covid-dataset/covid-positive/1109.jpg', cv2.IMREAD_UNCHANGED)
print('OpenCV: ', img.shape)