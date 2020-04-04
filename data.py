
import cv2

vc = cv2.VideoCapture('nixmas/nixmas/EXP_apu2_01_check-watch_cam4_frames_0112_0160.avi')
n = 1

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

timeF = 25

i = 0
while rval:
    rval, frame = vc.read()
    if (n % timeF == 0):  
        i += 1
        print(i)
        cv2.imwrite('trainingdata.jpg'.format(i), frame)
    n = n + 1
    cv2.waitKey(1)

