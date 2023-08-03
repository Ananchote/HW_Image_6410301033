import numpy as np
import cv2 as cv

r = 100
h = 300
k = 300
resolution = 500
sampling_line=500

imge = np.zeros([resolution,resolution], dtype=np.uint8)

for y in range(0,resolution):
    for x in range(0,resolution):
        if (x-h)**2 + (y-k)**2 > (r**2) and (x-h)**2 + (y-k)**2 < (r**2)+sampling_line:
            imge[x,y]=255
    
cv.imwrite("circle.png",imge)

cv.waitKey(0)
cv.destroyAllWindows()