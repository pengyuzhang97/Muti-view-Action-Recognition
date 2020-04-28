import cv2
import glob as glob
import numpy as np

matrix1 = glob.glob("D:/pictures/test/cam5/*avi*.jpg") #####################################

image = cv2.imread(matrix1[0],cv2.IMREAD_GRAYSCALE)
real_image = image[np.newaxis, :]

for i in range(1,len(matrix1)):
    image = cv2.imread(matrix1[i],cv2.IMREAD_GRAYSCALE)
    image = (image - image.mean())/image.std()
    image = image[np.newaxis,:]
    real_image = np.vstack((real_image,image))




np.save('test data from cam5',real_image) ######################################################




