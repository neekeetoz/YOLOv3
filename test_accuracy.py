import cv2
import numpy as np
from shapely.geometry import Polygon
from keras.models import load_model
from yolo import YOLO
from PIL import Image
import argparse

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

# загрузка модели YOLOv3 и конфигурации
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

parser = argparse.ArgumentParser("--input ")
args = {
    "image" : False,
    "input" : "images/0acd804f-79resized.jpg",
    "output" : ""
}
yolo = YOLO(False, "images/0acd804f-79resized.jpg", "")

# загрузка классов
classes = []
with open("input/signs_classes.txt", "r", encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

# загрузка изображений и разметки
images = []
gt_labels = []
with open("model_data/signs_annotations.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        image_path = line.rstrip().split(' ')[0]
        gt_label = line.rstrip().split(' ')[1:]
        images.append(cv2.imread(image_path))
        gt_labels.append([x.split(',') for x in gt_label])

yolo.detect_image(YOLO, images[0], [])
# расчет accuracy
total_correct = 0
total_objects = 0


for i in range(len(images)):
    image = images[i]
    gt_label = gt_labels[i]
    detect_img()
    
    
    
    (H, W) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(model.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    for j in range(len(gt_label)):
        (class_name, x, y, w, h) = gt_label[j].split(":")
        x = int(float(x) * W)
        y = int(float(y) * H)
        w = int(float(w) * W)
        h = int(float(h) * H)

        gt_box = [x, y, w, h]
        found = False

        for idx in idxs:
            if classIDs[idx[0]] == classes.index(class_name):
                box = boxes[idx[0]]
                iou = calculate_iou(gt_box, box)

                if iou >= 0.5:
                    total_correct += 1
                    found = True

        if not found:
            print("Object not detected:", class_name, "in image:", i)

    total_objects += len(gt_label)

accuracy = total_correct / total_objects
print("Accuracy:", accuracy)