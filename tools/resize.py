from PIL import Image
import os, sys, cv2

# путь к папке с изображениями
path = "C:/Users/M.Kalentyiev/Downloads/images/яндекс/2"
# путь к папке с результатами
out_path = "C:/Users/M.Kalentyiev/Downloads/images/яндекс/2-720"
# начало имени файла
f_name = "yandex_"
if not os.path.exists(out_path):
    os.makedirs(out_path)
dirs = os.listdir( path )

final_size = 720

def resize_aspect_fit():
    i = 1
    for item in dirs:
         if item == '.DS_Store':
             continue
         if os.path.isfile(path+'/'+item):
             im = Image.open(path+'/'+item)
             f, e = os.path.splitext(item)
             size = im.size
             ratio = float(final_size) / max(size)
             new_image_size = tuple([int(x*ratio) for x in size])
             im = im.resize(new_image_size, Image.ANTIALIAS)
             new_im = Image.new("RGB", (final_size, final_size))
             new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
             new_im.save(out_path + '/' + f_name + str(i) + '_720.jpg', 'JPEG', quality=95)
             i += 1

def resize_aspect(path):
    if os.path.isfile(path):
        im = Image.open(path)
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        im = im.resize(new_image_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        new_im.save(path, 'JPEG', quality=95)

def resize_images(paths, size):
    """
    Функция изменяет размер изображения, сохраняя его пропорции
    :param paths: Массив путей к изображениям
    :param size: Размер, который должен получиться. Например, (800, 600)
    """
    index = 0
    for path in paths:
        # Открываем изображение
        with Image.open(path) as image:
            # Получаем ширину и высоту оригинального изображения
            width, height = image.size
            # Вычисляем соотношение сторон
            aspect_ratio = width / height
            # Вычисляем новую ширину и высоту в соответствии с новым размером и соотношением сторон оригинального изображения
            if width < height:
                new_height = size[1]
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = size[0]
                new_height = int(new_width / aspect_ratio)
            # Изменяем размер изображения
            resized_image = image.resize((new_width, new_height))
            # Сохраняем измененное изображение
            filename, extension = os.path.splitext(path)
            resized_image.save(f"{out_path}/{f_name}{str(index)}_{size[0]}x{size[1]}{extension}")
            index+=1
     
def resize_annotations(annotation_file_path, annotation_path_out):
    """
    Изменяет размер изображения
    и обновляет данные в выделенных объектах
    
    Args:
        annotation_file_path (string): путь к аннотациям
    """
    with open(annotation_file_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            image_path = line[0]
            boxes = line[1:]
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
            new_height = 720
            new_width = 720
            resize_aspect(image_path)
            
            new_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax, class_id = map(int, box.split(','))
                k = new_width/img_width
                xmin = int(k * xmin)
                ymin = int((new_height - k * new_height) / 2 + ymin)
                xmax = int(k * xmax)
                ymax = int((new_height - k * new_height) / 2 + ymax)
                new_boxes.append([xmin, ymin, xmax, ymax, class_id])
            
            with open(annotation_path_out, 'a') as f_new:
                new_line = image_path + ' '
                for box in new_boxes:
                    box = ','.join(map(str, box))
                    new_line += box + ' '
                new_line = new_line.strip() + '\n'
                f_new.write(new_line)
                
            #os.remove(image_path)
            #cv2.imwrite(image_path, img)
            
resize_annotations('C:/Users/M.Kalentyiev/Documents/Python Scripts/from_github/yolo3/input/annotations_filtered.txt', 'C:/Users/M.Kalentyiev/Documents/Python Scripts/from_github/yolo3/input/annotations_416.txt')
# resize_aspect_fit()
# resize_images([(path + '/' + x) for x in dirs], (1280,720))