import colorsys
import os
from timeit import default_timer as timer
import time

import datetime
import cv2
from parser_gps import GPS
import parser_gps
import numpy as np
import tensorflow as tf
import math
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image


class YOLO(object):
    _defaults = {
        "model_path": 'input/trained_weights_final.h5',
        "anchors_path": 'model_data/signs_anchors.txt',
        "classes_path": 'input/signs_classes.txt',
        "score": 0.51,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
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
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
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
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            # from keras.utils import multi_gpu_model
            # self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
            session = tf.distribute.MirroredStrategy()
            with session.scope():
                model = load_model("model_data/yolo.h5")
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, centroids=[], is_enable=True):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
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
                    centroid.LifeTime = Centroid.DEFAULT_LIFETIME
                    is_exist = True
                    # запись индекса в метку
                    label_index = str(centroid.Index)
                    draw.rectangle([centroid.X, centroid.Y, centroid.X + 2, centroid.Y + 2], outline=self.colors[c])
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
                    (bottom - top) / 2 + top,
                    score,
                    (right - left)
                )
                centroid.LifeTime = Centroid.DEFAULT_LIFETIME
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
                draw.rectangle([centroid.X, centroid.Y, centroid.X + 2, centroid.Y + 2], outline=self.colors[c])
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
        if len(centroids) > 0:
            while True:
                if i >= len(centroids):
                    break
                centroids[i].LifeTime -= 1
                if centroids[i].LifeTime <= 0:
                    centroids.pop(i)
                else:
                    i += 1
        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


# проверка точности распознавания по датасету из картинок
# если центроид окажется внутри распознаваемого объекта
# и класс совпадает, то объект распознан верно
def test_accuracy(yolo):
    annotation_path = "model_data/signs_annotations.txt"
    classes_path = "input/signs_classes.txt"
    print("start testing...")
    # обнаруженные центроиды
    centroids = []
    # должно быть распознано объектов
    num_need_detect = 0
    # распознано правильно
    num_ok_detect = 0
    # распознано ложно
    num_wrong_detect = 0
    
    class_names = []
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
    
    with open(annotation_path) as f:
        lines = f.readlines()
    
    for line in lines:
        annotation = line.split(' ')
        img = Image.open(annotation[0].strip())
        num_need_detect += len(annotation) - 1
        yolo.detect_image(img, centroids, True)
        if len(annotation) > 1:
            num_wrong_detect += (len(annotation) - 1 + len(centroids))
            for c in centroids:
                box = annotation[1].strip().split(',')
                if c.compare(class_names[int(box[4])], int(box[0]), int(box[1]), int(box[2]), int(box[3])):
                    num_ok_detect += 1
                else:
                    num_wrong_detect += 1
        elif len(centroids) > 0:
            num_wrong_detect += len(centroids)
        centroids = []
    print('Правильно/Всего: ' + str(num_ok_detect) + '/' + str(num_need_detect) + ' Ошибочно: ' + str(num_wrong_detect))
    
    
      
        
def detect_video(yolo, video_path, output_path=""):
    # код для проверки точности обученной модели
    #test_accuracy(yolo)
    #yolo.close_session()
    #return
    # код без проверки
    gps: GPS = []
    gps = parser_gps.get_gps_data_from_file(video_path)
    try:
        time_start = datetime.datetime.strptime(gps[0].time, '%H:%M:%S')
    except:
        pass

    if video_path == '0':
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
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
    # это точки, указывающие на центр объекта, чтобы при обнаружении знать, что это тот же самый объект
    centroids = []
    # Номер кадра для проверки
    num_frame = 0
    num_gps = 0
    
    with open(video_path + '.csv', 'w') as csv_file:
        csv_file.write('id;name;score;latitude;longitude;date;time;speed\n')
        # данные объектов, записанные в файл
        written_centroids = []
        # объекты, обнаруженные в прошлом кадре
        last_centroids = []
        while True:
            return_value, frame = vid.read()
            if not return_value:
                print('End of file')
                break
            image = Image.fromarray(frame)
            #if num_frame <= 0:
            #    num_frame = 3
                # поиск объектов в кадре
            image = yolo.detect_image(image, centroids, True)
            #else:
            #    num_frame -= 1
            #    image = yolo.detect_image(image, centroids, False)
            
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
            cv2.putText(result, text=fps, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(158, 170, 6), thickness=3)

            time_frame = int(vid.get(cv2.CAP_PROP_POS_MSEC)/1000)
            try:
                if datetime.datetime.strptime(gps[num_gps].time, '%H:%M:%S') < (time_start + datetime.timedelta(seconds=time_frame)):
                    num_gps += 1
            except:
                pass
            # выходная строка с данными обнаруженного объекта
            out_line = ''
            if len(last_centroids) == 0:
                last_centroids = centroids.copy()
            # добавляем объект для записи в таблицу
            if len(last_centroids) > 0:
                for c in last_centroids:
                    # если объект был в прошлом кадре, но сейчас его нет,
                    # он вышел за пределы видимой области и находится там же, где и камера
                    # то есть машина проехала
                    if not c in written_centroids and not c in centroids:
                        out_line += str(c.Index) + ';' + c.Name + ';' + str(round(c.Score, 2))
                        
                        if gps[num_gps].latitude != None and gps[num_gps].longitude != None:
                            # todo: поменять местами широту и долготу
                            (sign_latitude, sign_longitude) = get_obj_coord(float(gps[num_gps].longitude), float(gps[num_gps].latitude), c.Size, get_side(image.size[0], c.X), image)
                            out_line += ';' + str(sign_latitude)
                            out_line += ';' + str(sign_longitude)
                        else:
                             out_line += ';;'
                        if gps[num_gps].date != None:
                            out_line += ';' + str(gps[num_gps].date)
                        else:
                             out_line += ';'
                        if gps[num_gps].time != None:
                            out_line += ';' + str(gps[num_gps].time)
                        else:
                             out_line += ';'
                        if gps[num_gps].speed != None:
                            out_line += ';' + str(gps[num_gps].speed)
                        else:
                             out_line += ';'
                        out_line += '\n'
                        written_centroids.append(c)
                last_centroids = centroids.copy()
            if out_line != '':
                csv_file.write(out_line)
            if gps[num_gps].latitude != None:
                last_latitude = gps[num_gps].latitude
                last_longitude = gps[num_gps].longitude
                cv2.putText(result, text='latitude: ' + str(gps[num_gps].latitude), org=(50, 100),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(158, 170, 6), thickness=3)
                out_line += ';' + str(gps[num_gps].longitude)
                cv2.putText(result, text='longitude: ' + str(gps[num_gps].longitude), org=(50, 150),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(158, 170, 6), thickness=3)
                out_line += ';' + str(gps[num_gps].date)
                cv2.putText(result, text='date: ' + str(gps[num_gps].date), org=(50, 250),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(158, 170, 6), thickness=3)
                out_line += ';' + str(gps[num_gps].time)
                cv2.putText(result, text='time: ' + str(gps[num_gps].time), org=(50, 300),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(158, 170, 6), thickness=3)
                out_line += ';' + str(gps[num_gps].speed)
                cv2.putText(result, text=f'speed: {str(gps[num_gps].speed)} km/h', org=(50, 200),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(158, 170, 6), thickness=3)
                
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out_line = ''
                for c in centroids:
                    # записываем все обнаруженные объекты, так как обнаружен конец видео
                    if not c in written_centroids:
                        out_line += str(c.Index) + ';' + c.Name + ';' + str(round(c.Score, 2))
                        if gps[num_gps].latitude != None:
                            out_line += ';' + str(gps[num_gps].latitude)
                        else:
                             out_line += ';'
                        if gps[num_gps].longitude != None:
                            out_line += ';' + str(gps[num_gps].longitude)
                        else:
                             out_line += ';'
                        if gps[num_gps].date != None:
                            out_line += ';' + str(gps[num_gps].date)
                        else:
                             out_line += ';'
                        if gps[num_gps].time != None:
                            out_line += ';' + str(gps[num_gps].time)
                        else:
                             out_line += ';'
                        if gps[num_gps].speed != None:
                            out_line += ';' + str(gps[num_gps].speed)
                        else:
                             out_line += ';'
                        out_line += '\n'
                        written_centroids.append(c)
                csv_file.write(out_line)
                break
        yolo.close_session()


def get_side(width, x):
    """возвращает сторону на которой находится координата

    Args:
        width (double): ширина изображения
        x (double): координата X

    Returns:
        str: сторона
    """
    return x >= (width / 2)
    
last_latitude = 0
last_longitude = 0
def get_obj_coord(latitude, longitude, width, isRight, image):
    """Возвращает координаты объекта на изображении, относительно камеры

    Args:
        latitude (double): широта
        longitude (double): долгота
        width (int): ширина рамки обнаруженного объекта в px
        side (SideImage, optional): с какой стороны находится объект.

    Returns:
        (double, double): широта и долгота
    """
    # стандартные размеры знака в пикселях
    default_width = 100
    default_height = 100
    # определяем направление
    isNorth = ((latitude - last_latitude) >= 0)
    isSouth = ((latitude - last_latitude) <= 0)
    isEast = ((longitude - last_longitude) >= 0)
    isWest = ((longitude - last_longitude) <= 0)
    
    delim = int(longitude/10)
    multiple_latitude = 71.7
    # у каждой широты 1 градус имеет свое расстояние в км
    match delim:
        case 4:
            multiple_latitude = 85.4
        case 5:
            multiple_latitude = 71.7
        case 6:
            multiple_latitude = 55.8
        case 7:
            multiple_latitude = 38.2
    multiple_longitude = 111.1
    
    # реальная ширина знака в метрах
    width_real = 0.6
    # real_height = 0.6   # высота объекта в реальности, метры
    # image_height = 56   # высота объекта на фото, пиксели
    # distance = 5        # расстояние от камеры до объекта, метры
    # focal_length = (image_height * distance) / real_height
    focal_length = 560
    # расстояние от камеры до объекта
    #distance_forward = width_real / width * 6
    distance_forward = (width_real * focal_length) / width
    # фокусное расстояние
    f = 1000
    
    angle = math.radians(170)
    k = 2 * math.tan(angle/2) / image.size[0]
    # расстояние в сторону от камеры в метрах
    distance_to_side = k * focal_length / (1 - k * distance_forward)
    distance_to_side *= -1 if isRight else 1
    # todo: здесь возможна ошибка в выборе знака (+/-)
    if isNorth:
        longitude += distance_to_side / 1000 / multiple_longitude
    if isSouth:
        longitude -= distance_to_side / 1000 / multiple_longitude
    if isEast:
        latitude -= distance_forward / 1000 / multiple_latitude
    if isWest:
        latitude += distance_forward / 1000 / multiple_latitude
    return (round(latitude, 6), round(longitude, 6))
        

# класс хранит координаты центральной точки обнаруженного объекта
class Centroid:
    # время существования центроида (количество кадров) по умолчанию
    DEFAULT_LIFETIME = 6
    # для всех объектов
    index = 0

    def __init__(self, name, x, y, score, size):
        self.Name = name
        self.X = x
        self.Y = y
        self.Score = score
        self.Index = Centroid.index
        # время существования центроида (количество кадров)
        self.LifeTime = self.DEFAULT_LIFETIME
        # ширина или высота объекта
        self.Size = size

    # Проверяет является ли центроид этим объектом
    def compare(self, name, x1, y1, x2, y2):
        if self.Name == name:
            if self.X > x1 and self.X < x2:
                if self.Y > y1 and self.Y < y2:
                    return True
        return False