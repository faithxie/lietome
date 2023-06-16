import cv2
import numpy as np

# 加载YOLOv4配置和权重文件
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

# 加载类别标签
# classes = ['person', 'bicycle','car']
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 加载图像
image = cv2.imread('1.jpg')
height, width, channels = image.shape

# 创建blob对象
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# 设置网络的输入
net.setInput(blob)

# 前向传播，获取输出层
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# outs = net.forward(output_layers)


layer_names = net.getLayerNames()
output_layers = list(set([layer_names[i - 1] for i in net.getUnconnectedOutLayers()]))
outs = net.forward(output_layers)


class_ids = []
confidences = []
boxes = []

# 初始化计数器
person_count = 0

# 解析网络输出，获取检测结果
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if classes[class_id] == "person" and confidence > 0.7:
        # if confidence > 0.5:  # 设置置信度阈值
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

            person_count += 1

# 绘制边界框和标签
font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    # label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    # print(label)
    color = (0, 255, 0)  # 绿色边界框
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), font, 0.5, color, 2)

# 显示结果
print("人数：", person_count)
# 显示结果图像
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
