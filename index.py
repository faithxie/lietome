import cv2
import numpy as np
import time
import random

import face_recognition
# url = "rtsp://admin:doposoft123@192.168.10.1:554/h264/ch1/main/av_stream"
# url = "http://hls01open.ys7.com/openlive/bd793ee3abd54f64a04116c5478b17c3.hd.m3u8"
# url = 'hr.mkv'
url = '1.mp4'
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)  # 0为电脑内置摄像头  1为电脑外置摄像头
# cap = cv2.VideoCapture(1)  #
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
# cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

#以下是读取视频流截取图片的方法

filename = 'image/1.jpg'
while True:
    (ret, frame) = cap.read()
    # frame = cv2.flip(frame, 0) //垂直翻转
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    #rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(628, 628))
    # rects = cascade.detectMultiScale(gray_frame, 1.1,3,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))


    if len(rects) > 0:  # 如果>0说明检测到人了
        print(rects )
        # 对比前一张
        pre_filename = filename

        filename = 'FaceImgReg/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + random.choice(
            'abcdefghijklmnopqrstuvwxyz') + '.jpg'
        cv2.imwrite(filename, frame)
        time.sleep(3)

        print(pre_filename)
        print(filename)
        '''
        first_image = face_recognition.load_image_file(pre_filename)
        second_image = face_recognition.load_image_file(filename)

        first_faceNum = face_recognition.face_encodings(first_image)
        second_faceNum = face_recognition.face_encodings(second_image)
        if (len(first_faceNum)>0 and len(second_faceNum)>0):
            # 获取检测到人脸时面部编码信息中第一个面部编码
            first_encoding = face_recognition.face_encodings(first_image)[0]
            second_encoding = face_recognition.face_encodings(second_image)[0]

            results = face_recognition.compare_faces([first_encoding], second_encoding)
            print(results[0])
            if results[0]:
                print(results)
            else:
                savefilename = 'FaceImgRegSave1/' + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                 time.localtime()) + random.choice(
                    'abcdefghijklmnopqrstuvwxyz') + '.jpg'
                cv2.imwrite(savefilename, frame)
                cv2.imshow("Video", frame)
        else:
            print( len(first_faceNum))
            print( len(second_faceNum))
            print(len(first_faceNum)>0 and len(second_faceNum))
        '''
        #带上红框定位
        for rect in rects:
            x, y, w, h = rect
            p1, p2 = (x, y), (x + w, y + h)
            cv2.rectangle(frame, p1, p2, color=(0, 0, 255), thickness=2)

    else:
        print(rects)
        print('runnning......')
    cv2.imshow("Video", frame)
    cv2.waitKey(1)
'''

#以下是人脸比对的方法
# first_image = face_recognition.load_image_file("image/2020-10-08-16-57-14d.jpg")
# second_image = face_recognition.load_image_file("image/2020-10-08-17-03-05a.jpg")

first_image = face_recognition.load_image_file("FaceImgReg/2021-01-09-10-29-04g.jpg")
second_image = face_recognition.load_image_file("FaceImgReg/2021-01-09-10-28-47t.jpg")

first_encoding = face_recognition.face_encodings(first_image)[0]
second_encoding = face_recognition.face_encodings(second_image)[0]

results = face_recognition.compare_faces([first_encoding], second_encoding)
print(results)
'''