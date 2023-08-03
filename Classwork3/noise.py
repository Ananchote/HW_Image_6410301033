import cv2 as cv 
import random as rd
import numpy as np

catt = "./cat.jpeg"

img = cv.imread(catt,cv.IMREAD_GRAYSCALE)
img = np.array(img,dtype="uint8")

density_salt = 0.1
density_pepper = 0.1

number_of_white_pexel = int(density_salt * img.shape[0] * img.shape[1])
number_of_black_pexel = int(density_pepper * img.shape[0] * img.shape[1])
noies=np.array(img,dtype="uint8")

for i in range(number_of_white_pexel):
    y=rd.randint(0,img.shape[0]-1)
    x=rd.randint(0,img.shape[1]-1)
    img[y][x]=255
    
for i in range(number_of_white_pexel):
    y=rd.randint(0,img.shape[0]-1)
    x=rd.randint(0,img.shape[1]-1)
    img[y][x]=0
    
cv.imwrite("./noised.png",img)

denoised = cv.medianBlur(noies, 7)

cv.imwrite("./denoised.png", denoised)

out = denoised-img
cv.imwrite("./noised.png",noies)
cv.imwrite("./ori.png",img)
cv.imwrite("./out.png",out)
