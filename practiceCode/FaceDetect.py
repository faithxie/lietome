# 1.通过 url或摄像头获取视频数据
# 2.detectMultiScale 侦测人脸
# 3.保存到 FaceImgReg

import cv2
import numpy as np
import time
import random

import face_recognition
# url = "rtsp://admin:doposoft123@192.168.10.1:554/h264/ch1/main/av_stream"
# url = "http://hls01open.ys7.com/openlive/bd793ee3abd54f64a04116c5478b17c3.hd.m3u8"
# url = 'hr.mkv'
#url = '1.mp4'
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


    '''
    void detectMultiScale(
        const Mat& image,
        CV_OUT vector<Rect>& objects,
        double scaleFactor = 1.1,
        int minNeighbors = 3, 
        int flags = 0,
        Size minSize = Size(),
        Size maxSize = Size()
    );
    
    参数1：image--待检测图片，一般为灰度图像加快检测速度；
    参数2：objects--被检测物体的矩形框向量组；
    参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
    参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
            如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
            如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
            这种设定值一般用在用户自定义对检测结果的组合程序上；
    参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
            CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
            因此这些区域通常不会是人脸所在区域；
    参数6、7：minSize和maxSize用来限制得到的目标区域的范围。

    '''
    rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    #rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(628, 628))
    # rects = cascade.detectMultiScale(gray_frame, 1.1,3,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))


    if len(rects) > 0:  # 如果>0说明检测到人了
        print(rects)
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