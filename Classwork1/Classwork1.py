#Ananchote   6410301033
import cv2 as cv
import numpy as np

image=np.zeros([50,50])

start_point = (15,15)

end_point = (35,35)

color = (200, 200 ,200)

thickness = 1

image =  cv.line(image,start_point,end_point,color,thickness)

cv.imwrite("./classwork1.png",image)
