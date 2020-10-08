import cv2
import numpy as np
import time
import random

# url = "rtsp://admin:doposoft123@192.168.10.1:554/h264/ch1/main/av_stream"
# url = "http://hls01open.ys7.com/openlive/bd793ee3abd54f64a04116c5478b17c3.hd.m3u8"
# url = 'hr.mkv'
url = 'Lie.To.Me.2009.S01E01.BD1080P.X264.AAC.English.CHS-ENG.BDE4.mp4'
cap = cv2.VideoCapture(url)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
# cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
while True:
    (ret, frame) = cap.read()
    # frame = cv2.flip(frame, 0) //垂直翻转
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(328, 328))
    if len(rects) > 0:  # 如果>0说明检测到人了
        # print(rects )
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        for rect in rects:
            x, y, w, h = rect
            p1, p2 = (x, y), (x + w, y + h)
            cv2.rectangle(frame, p1, p2, color=(0, 0, 255), thickness=2)
            filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + random.choice(
            'abcdefghijklmnopqrstuvwxyz') + '.jpg'
            # cv2.imwrite(filename, frame)
            cv2.imshow("Video", frame)
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'......')
    # cv2.imshow("Video", frame)
    cv2.waitKey(1)
#以下是最常用的读取视频流的方法
# cap = cv2.VideoCapture(url)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()