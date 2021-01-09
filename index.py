import cv2
import numpy as np
import time
import random

import face_recognition
# url = "rtsp://admin:doposoft123@192.168.10.1:554/h264/ch1/main/av_stream"
# url = "http://hls01open.ys7.com/openlive/bd793ee3abd54f64a04116c5478b17c3.hd.m3u8"
# url = 'hr.mkv'
url = 'Lie.To.Me.2009.S01E01.BD1080P.X264.AAC.English.CHS-ENG.BDE4.mp4'
cap = cv2.VideoCapture(url)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
# cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

#以下是读取视频流截取图片的方法

while True:
    (ret, frame) = cap.read()
    # frame = cv2.flip(frame, 0) //垂直翻转
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    # rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(628, 628))
    rects = cascade.detectMultiScale(gray_frame, 1.1,3,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))

    if len(rects) > 0:  # 如果>0说明检测到人了
        print(rects )
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        for rect in rects:
            x, y, w, h = rect
            p1, p2 = (x, y), (x + w, y + h)
            cv2.rectangle(frame, p1, p2, color=(0, 0, 255), thickness=2)
            filename = 'FaceImg/'+ time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + random.choice(
            'abcdefghijklmnopqrstuvwxyz') + '.jpg'
            cv2.imwrite(filename, frame)
            cv2.imshow("Video", frame)
    else:
        print(rects)
        print('runnning......')
    # cv2.imshow("Video", frame)
    cv2.waitKey(1)


#以下是人脸比对的方法
# first_image = face_recognition.load_image_file("image/2020-10-08-16-57-14d.jpg")
# second_image = face_recognition.load_image_file("image/2020-10-08-17-03-05a.jpg")

# first_image = face_recognition.load_image_file("image/1.jpg")
# second_image = face_recognition.load_image_file("image/2.jpg")
#
# first_encoding = face_recognition.face_encodings(first_image)[0]
# second_encoding = face_recognition.face_encodings(second_image)[0]

results = face_recognition.compare_faces([first_encoding], second_encoding)
print(results)