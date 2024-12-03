import cv2 as cv
import os

path = os.getcwd()

print("******************************************")
print(f" Carpeta actual: {os.getcwd()}")
print("******************************************")

img = cv.imread(path+'/flor.jpg')
cv.imshow('flor', img)

cv.waitKey(0)

cv.destroyAllWindows()