import cv2 as cv
import numpy as np

image=cv.imread("123.jpg",cv.IMREAD_GRAYSCALE)
kernnel =cv.imread("classwork1.png",cv.IMREAD_GRAYSCALE)
n = kernnel.sum()
kernnel = kernnel/n

image = cv.filter2D(src=image, ddepth=-1, kernel=kernnel)

cv.imwrite(".456.png",image)