# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Предисловие

Реализация Keras для YOLOv3 (серверная часть Tensorflow), вдохновленная [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K ).


---

## БЫСТРЫЙ СТАРТ

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.
1. Скачайте веса YOLOv3 с [веб-сайта YOLO](http://pjreddie.com/darknet/yolo /).
2. Преобразуйте модель Darknet YOLO в модель Keras.
3. Запустите YOLO обнаружение.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

Для Tiny YOLOv3 аналогичное, но с указанием другой модели и якорей `--model model_file` and `--anchors anchor_file`.

### ИСПОЛЬЗОВАНИЕ 
Для того, чтобы получить подсказку по использованию yolo_video.py введите команду --help:
```
запускаемый файл с параметрами: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

аргументы для указания пути к видео:
  --input        Путь к исходному видео
  --output       Путь к выходному видео (необязательно)

необязательные аргументы:
  -h, --help         помощь, справка
  --model MODEL      путь к файлу с весами, по умолчанию model_data/yolo.h5
  --anchors ANCHORS  путь к файлу с якорями (привязка к координатам, областям интересов), по умолчанию
                     model_data/yolo_anchors.txt
  --classes CLASSES  путь к файлу с классами, по умолчанию
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  количество используемого ГПУ, по умолчанию 1
  --image            Обнаружить на картинке вместо видео
```
---

4. Для использования нескольких ГПУ (видеокарт): используйте `--gpu_num N` где N число графических процессоров. Передается в [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Обучение

1. Создайте свой файл аннотаций (с помощью CVT или Label Studio) и файл с классами.  
    Пример строки аннотаций;  
    Строка: `image_file_path box1 box2 ... boxN`;  
    Коробка: `x_min,y_min,x_max,y_max,class_id` (без пробелов).  
    Для набора данных VOC: `python voc_annotation.py`  
    Пример:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Убедитесь, что вы запустили `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`
   Файл model_data/yolo_weights.h5 используется для загрузки предварительно подготовленных весов.

3. Измените train.py и начните обучение.  
    `python train.py`  
    Используйте свои тренировочные веса или контрольные веса с помощью опции командной строки`--model model_file` когда используте yolo_video.py
    Не забудьте изменить путь к классу или путь привязки с помощью `--classes class_file` и `--anchors anchor_file`.

Для получению оригинальных весов YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. Переименуйте darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. Используйте model_data/darknet53_weights.h5 в train.py

---

## ДОПОЛНИТЕЛЬНО 

1. Версия Python и необходимые версии фреймворков.
    - Python 3.10.4
    - Keras 2.10.0
    - tensorflow 2.11.0

2. Для формирования своих якорей необходимо использовать convert.py.
