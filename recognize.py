import cv2
import os
# coding=utf-8
import time

#加载训练数据集文件
recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('FaceImgReg/trainer/trainer.yml')
names=[]
warningtime = 0


#准备识别的图片
def face_detect_demo(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    face_detector=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    # face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    face=face_detector.detectMultiScale(gray,1.2,3,cv2.CASCADE_SCALE_IMAGE,(32,32),(300,300))
    #face=face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        print('标签id:',ids,'置信评分：', confidence)
        if confidence < 80:
            global warningtime
            warningtime += 1
            if warningtime > 100:
               # warning()
               warningtime = 0
            cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:

            # cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            cv2.putText(img,str(ids), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            print('recog')
    cv2.imshow('result',img)

    # print('bug:',ids)

def name():
    path = 'FaceImgReg/identify/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[0])
       names.append(name)
    print(names)

cap = cv2.VideoCapture(0)  # 0为电脑内置摄像头  1为电脑外置摄像头
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    # time.sleep(1)
    resiez_img = cv2.resize(frame, dsize=(200, 200))
    face_detect_demo(resiez_img)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()
#print(names)
