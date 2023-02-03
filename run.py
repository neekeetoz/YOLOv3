import xml.etree.ElementTree as ET
import io
import os

def convert_data(img, xml_path, out_file, list_classes):
    str_out = img
    if (xml_path != ''):
        tree = ET.parse(xml_path)
        str_out = tree.find('filename').text
        for tag in tree.findall('object'):
            str_out += ' ' + tag.find('bndbox/xmin').text
            str_out += ',' + tag.find('bndbox/ymin').text
            str_out += ',' + tag.find('bndbox/xmax').text
            str_out += ',' + tag.find('bndbox/ymax').text
            str_out += ',' + str(list_classes.index(tag.find('name').text))
    with open(out_file, 'a') as f:
        f.write('images/' + str_out + '\n')

f_classes = open('input/classes_27.txt', 'r')
list_classes = f_classes.read().split('\n')
f_classes.close()
catalog_annotations = 'Annotations/'
for img in os.listdir('images'):
    img_name, _ = os.path.splitext(img)
    xml_path = ''
    if (os.path.exists(catalog_annotations + img_name + '.xml')):
         xml_path = catalog_annotations + img_name + '.xml'
    convert_data('images/' + img, xml_path, 'input/train_27c.txt', list_classes)