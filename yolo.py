# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import pandas as pd
import colorsys
import os
from timeit import default_timer as timer
import time

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw, ExifTags

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

class YOLO(object):
    _defaults = {
        "model_path": 'input/deafult_weigths.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        # последние обнаруженные объекты
        self.out_boxes = None
        self.out_scores = None
        self.out_classes = None

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            #from keras.utils import multi_gpu_model
            #self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
            import tensorflow as tf
            session = tf.distribute.MirroredStrategy()
            with session.scope():
                model = load_model("model_data/yolo.h5")
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, centroids, is_enable):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        if is_enable:
            self.out_boxes, self.out_scores, self.out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    # size[1] - ширина, size[0] - высота
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
        

        print('Found {} boxes for {}'.format(len(self.out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # сбрасываем параметр на случай, если центроид больше не сущуствует
        for centroid in centroids:
            centroid.Exist = False
        
        for i, c in reversed(list(enumerate(self.out_classes))):
            # распознанный класс
            predicted_class = self.class_names[c]
            box = self.out_boxes[i]
            score = self.out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            
            draw = ImageDraw.Draw(image)

            # координаты
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            # если не существуют, то удалить, иначе изменить центральную координату
            is_exist = False
            for centroid in centroids:
                if centroid.compare(predicted_class, left, top, right, bottom):
                    centroid.X = (right - left) / 2 + left
                    centroid.Y = (bottom - top) / 2 + top
                    centroid.Exist = True
                    is_exist = True
                    # запись индекса в метку
                    label_index = str(centroid.Index)
                    draw.rectangle([centroid.X, centroid.Y, centroid.X+2, centroid.Y+2], outline=self.colors[c])
                    label_size = draw.textsize(label, font)
                    if top - label_size[1] >= 0:
                        text_origin = np.array([centroid.X, centroid.Y - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])
                    draw.text(text_origin, label_index, fill=(0, 256, 0), font=font)
                    break
            # Если центроида для найденного объекта не существует, то создаем новый
            if not is_exist:
                centroid = Centroid(
                    predicted_class,
                    (right - left) / 2 + left,
                    (bottom - top) / 2 + top
                )
                centroid.Exist = True
                centroids.append(centroid)
                # запись индекса в метку
                label_index = str(centroid.Index)
                # Индекс для следующего объекта
                Centroid.index += 1
                label_size = draw.textsize(label, font)
                if top - label_size[1] >= 0:
                    text_origin = np.array([centroid.X, centroid.Y - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                # Рисуем точку центроида и номер объекта
                draw.rectangle([centroid.X, centroid.Y, centroid.X+2, centroid.Y+2], outline=self.colors[c])
                draw.text(text_origin, label_index, fill=(0, 256, 0), font=font)

            print(label, label_index, (left, top), (right, bottom))
            label_size = draw.textsize(label, font)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                # рисуем
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # del draw
            
        # удаляем неактуальные центроиды
        i = 0
        while True:
            if i >= len(centroids):
                break
            if not centroids[i].Exist:
                centroids.pop(i)
            i += 1
                
        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    if video_path == '0':
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    # ОБЪЕКТЫ ДЛЯ СРАВНЕНИЯ
    centroids = []
    # Номер кадра для проверки
    num_frame = 0
    # обработать видео в реальном времени или перед показом
    in_real_time = False
    if not in_real_time:
        result = []
    while True:
        return_value, frame = vid.read()
        if not return_value:
            print('End of file')
            break
        image = Image.fromarray(frame)
        # coordinates(image)
        if num_frame <= 0:
            num_frame = 5
            # поиск объектов в кадре
            image = yolo.detect_image(image, centroids, True)
        else:
            num_frame -= 1
            image = yolo.detect_image(image, centroids, False)
        if in_real_time:
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
        else:
            result.append(np.asarray(image))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not in_real_time:
        prev_time = timer()
        count_frames = 60
        delay = 1 / count_frames
        for fr in result:
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(fr, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", fr)
            if isOutput:
                out.write(fr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(delay)
    yolo.close_session()
    
    
def coordinates(image):
    result = [0] * 2 
    j=0
    exif = { ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS }
    gps = exif['GPSInfo']
        
    result[j] = gps[2][0] + gps[2][1]/60 + float(gps[2][2])/3600
    result[j+1] = gps[4][0] + gps[4][1]/60 + float(gps[4][2])/3600

    # coord[j] = str(int(gps[2][0])) + '°' + str(int(gps[2][1])) + '`' + str(int(gps[2][2]))
    # coord[j+1] = str(int(gps[4][0])) + '°' + str(int(gps[4][1])) + '`' + str(int(gps[4][2]))
    j += 2

# МОЙ КЛАСС
class Centroid:
    # для всех объектов
    index = 0
    def __init__(self, name, x, y):
        self.Name = name
        self.X = x
        self.Y = y
        self.Index = Centroid.index
        self.Exist = True
        
    # Проверяет является ли центроид этим объектом
    def compare(self, name, x1, y1, x2, y2):
        if self.Name == name:
            if self.X > x1 and self.X < x2:
                if self.Y > y1 and self.Y < y2:
                    return True
        return False
    