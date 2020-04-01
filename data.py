
import cv2

vc = cv2.VideoCapture('nixmas/nixmas/EXP_apu2_01_check-watch_cam4_frames_0112_0160.avi')  # 读入视频文件，命名cv
n = 1  # 计数

if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False

timeF = 25  # 视频帧计数间隔频率

i = 0
while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    if (n % timeF == 0):  # 每隔timeF帧进行存储操作
        i += 1
        print(i)
        cv2.imwrite('trainingdata.jpg'.format(i), frame)  # 存储为图像
    n = n + 1
    cv2.waitKey(1)

