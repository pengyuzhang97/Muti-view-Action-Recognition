import cv2
import glob as glob
import numpy as np

file1 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/Project/trainingdata/traindata all 5 cameras/*cam1*.avi")
file2 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/Project/trainingdata/traindata all 5 cameras/*cam2*.avi")
file3 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/Project/trainingdata/traindata all 5 cameras/*cam3*.avi")
file4 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/Project/trainingdata/traindata all 5 cameras/*cam4*.avi")
file5 = glob.glob("D:/UFL/OneDrive - University of Florida/Pattern Recognition/Project/trainingdata/traindata all 5 cameras/*cam5*.avi")

labels1_ = np.repeat(np.arange(0,11),11)
count = 0
count_value = 0
delete1 = np.zeros(28,int)
number1 = 0

for j in range(len(file5)):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(file5[j])
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
                # cv2.imwrite("D:/UFL/OneDrive - University of Florida/Pattern Recognition/Project/trainingdata/cam1/1-%d.avi(%d).jpg"%(j+1,i),frame)
                cv2.imwrite("D:\pictures/cam5/1-%d.avi(%d).jpg" % (j + 1, i), frame)

labels1 = np.delete(labels1_,delete1)
np.save('labels from cam5',labels1)