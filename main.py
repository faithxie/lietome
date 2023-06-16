import cv2
import time
import random

cap = cv2.VideoCapture(0)  # 0为电脑内置摄像头  1为电脑外置摄像头
# url = '1.mkv'
# cap = cv2.VideoCapture(url)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (ret, frame) = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(rects) > 0:  # 如果>0说明检测到人了
        print(rects)
        # 对比前一张
        filename = 'FaceImgReg/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + random.choice(
            'abcdefghijklmnopqrstuvwxyz') + '.jpg'

        # 带上红框定位
        for rect in rects:
            x, y, w, h = rect
            p1, p2 = (x, y), (x + w, y + h)
            cv2.rectangle(frame, p1, p2, color=(0, 0, 255), thickness=2)


        # 修改尺寸
        resize_img = cv2.resize(frame, dsize=(200, 200))
        #保存图片
        # cv2.imwrite(filename, resize_img)
        # print(filename)
        # time.sleep(3)

    else:
        print(rects)
        print('runnning......')
    cv2.imshow("Video", frame)
    cv2.waitKey(1)

#释放摄像头
cap.release()
#释放内存
cv2.destroyAllWindows()