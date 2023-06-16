import cv2
import numpy as np

# 加载YOLOv4配置文件和权重
net = cv2.dnn.readNet("yolov4.cfg", "yolov4.weights")

# 获取YOLOv4输出层的名称
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 加载图像
image = cv2.imread("1.jpg")
height, width, channels = image.shape

# 对图像进行预处理
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# 将预处理后的图像输入到网络中进行推理
net.setInput(blob)
outs = net.forward(output_layers)

# 解析YOLOv4输出结果
class_ids = []
confidences = []
boxes = []
person_count = 0

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.7 and class_id == 0:  # 确认为人的检测结果
            person_count += 1
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 绘制边界框和计数
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, 'Person', (x, y - 10), font, 0.5, (0, 255, 0), 2)

cv2.putText(image, f'Person count: {person_count}', (10, 30), font, 0.8, (0, 0, 255), 2)
cv2.imshow("YOLOv4 Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()