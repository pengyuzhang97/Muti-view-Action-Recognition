import cv2
import glob as glob
import numpy as np

file1 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/testingdataset/*cam1*.avi")
file2 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/testingdataset/*cam2*.avi")
file3 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/testingdataset/*cam3*.avi")
file4 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/testingdataset/*cam4*.avi")
file5 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/testingdataset/*cam5*.avi")

labels1_ = np.tile(np.arange(0,11),4)
count = 0
count_value = 0
delete1 = np.zeros(8,int)
number1 = 0

for j in range(len(file5)):  ####################################################
    videoCapture = cv2.VideoCapture()
    videoCapture.open(file5[j])  ###########################################################
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    #print("fps=",fps,"frames=",frames,)

    if frames<50:
        count = count+1
        delete1[number1] = j
        number1 = number1+1
        continue
    elif frames>=50:
        count_value +=1
        print("value=",count_value )
        for i in range(1, int(frames) + 1):
            if i <= 50:
                ret, frame = videoCapture.read()
                cv2.imwrite("D:/pictures/test/cam5/1-%d.avi(%d).jpg" % (j + 1, i), frame) #######################################################

labels1 = np.delete(labels1_,delete1)
np.save('test labels from cam5',labels1) #######################################