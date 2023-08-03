import numpy as np
import cv2 as cv

img = cv.imread('doc.jpg',cv.IMREAD_GRAYSCALE)

laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)

print('[Input]type',img.dtype)
print('[Laplacian] type',laplacian.dtype)
print('[Soble X] type',sobelx.dtype)

laplacian = cv.normalize(laplacian,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
sobelx =  cv.normalize(sobelx,None,0,255,cv.NORM_MINMAX,cv.CV_8U)

cv.imwrite('Laplacian.png',laplacian)
cv.imwrite('Sobelx.png',sobelx)

img = img.astype(np.float32)
img2 = sobelx.astype(np.float32)

imgF = np.fft.fft2(img) #take fourier transform
imgF2 = np.fft.fft2(img2)

imgF = np.fft.fftshift(imgF) #ship 0,0 to center of img
imgF2 = np.fft.fftshift(imgF2)

imgReal = np.real(imgF)
imgIma = np.imag(imgF)
imgMag = np.sqrt(imgReal**2 + imgIma**2)
imgPhs = np.arctan2(imgIma, imgReal)

imgReal2 = np.real(imgF2)
imgIma2 = np.imag(imgF2)
imgMag2 = np.sqrt(imgReal2**2 + imgIma2**2)
imgPhs2 = np.arctan2(imgIma2, imgReal2)

imgRealInv = imgMag * np.cos(imgPhs)
imgImaInv = imgMag*np.sin(imgPhs)

imgRealInv2 = imgMag2 * np.cos(imgPhs2)
imgImaInv2 = imgMag2*np.sin(imgPhs2)

imgFInv = imgRealInv + imgImaInv*1j
imgFInv = np.fft.ifftshift(imgFInv)
imgInv = np.fft.ifft2(imgFInv)

imgFInv2 = imgRealInv2 + imgImaInv2*1j
imgFInv2 = np.fft.ifftshift(imgFInv2)
imgInv2 = np.fft.ifft2(imgFInv2)

imgInv = np.real(imgInv)
imgInv = imgInv.astype(np.uint8)

imgInv2 = np.real(imgInv2)
imgInv2 = imgInv2.astype(np.uint8)

cv.imwrite('input_Frequency.png', img)
cv.imwrite('output_Frequency.png', imgInv)

cv.imwrite('output_SobelX.png', imgInv2)

imgMag = np.log(1+imgMag)
imgMag = cv.normalize(imgMag, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
cv.imwrite('mag_Filter.png',imgMag)

imgMag2 = np.log(1+imgMag2)
imgMag2 = cv.normalize(imgMag2, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
cv.imwrite('mag_Filter_SobelX.png',imgMag2)

sobel_x_magnitude = np.abs(imgMag2)
sobel_y_magnitude = np.abs(img)

filtered_image_x_freq_shifted = imgFInv * sobel_x_magnitude
filtered_image_y_freq_shifted = imgFInv * sobel_y_magnitude

filtered_image_x = np.fft.ifftshift(filtered_image_x_freq_shifted)
filtered_image_y = np.fft.ifftshift(filtered_image_y_freq_shifted)

filtered_image_x = np.fft.ifft2(filtered_image_x)
filtered_image_y = np.fft.ifft2(filtered_image_y)

filtered_image_x = np.abs(filtered_image_x)
filtered_image_y = np.abs(filtered_image_y)

filtered_image_x = np.uint8(filtered_image_x)
filtered_image_y = np.uint8(filtered_image_y)

cv.imshow("input image in gray scale",img)

cv.imshow("image_magnitude",imgMag2)

cv.imshow("magnitude_sobelx",sobel_x_magnitude)

cv.imshow("magnitude_sobely",sobel_y_magnitude)

cv.imshow('Sobel X', filtered_image_x)

cv.imshow('Sobel Y', filtered_image_y)
cv.waitKey(0)
cv.destroyAllWindows()