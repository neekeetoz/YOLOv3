import xml.etree.ElementTree as ET
import io
import os

def convert_data(img, xml_path, out_file, list_classes):
    str_out = img
    if (xml_path != ''):
        tree = ET.parse(xml_path)
        # str_out = tree.find('filename').text
        for tag in tree.findall('object'):
            str_out += ' ' + tag.find('bndbox/xmin').text
            str_out += ',' + tag.find('bndbox/ymin').text
            str_out += ',' + tag.find('bndbox/xmax').text
            str_out += ',' + tag.find('bndbox/ymax').text
            name = tag.find('name').text
            if name not in list_classes:
                list_classes.append(name)
            str_out += ',' + str(list_classes.index(name))
    with open(out_file, 'a') as f:
        f.write(str_out + '\n')


# программа объединяет файлы с аннотациями в один и создает файл для списка классов 
# здесь нужно указать свои пути
# путь к файлу с классами
path_to_classes_file = 'input/signs_classes_720p.txt'
# путь к каталогу с аннотациями
path_to_annotations = 'Annotations/'
# путь к каталогу с изображениями
path_to_images = 'images'
# выходной файл с анннотациями для обучения
path_to_out_annotations = 'model_data/signs_annotations_720p.txt'

list_classes = []
#f_classes.close()
for img in os.listdir(path_to_images):
    img_name, _ = os.path.splitext(img)
    xml_path = ''
    if (os.path.exists(path_to_annotations + img_name + '.xml')):
         xml_path = path_to_annotations + img_name + '.xml'
    convert_data(path_to_images + '/' + img, xml_path, path_to_out_annotations, list_classes)
f_classes = open(path_to_classes_file, 'w+', encoding='utf-8')
f_classes.write('\n'.join(list_classes))
f_classes.close()