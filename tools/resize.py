from PIL import Image
import os, sys

path = "C:/Users/M.Kalentyiev/Downloads/images/яндекс/2"
out_path = "C:/Users/M.Kalentyiev/Downloads/images/яндекс/2-720"
if not os.path.exists(out_path):
    os.makedirs(out_path)
dirs = os.listdir( path )

final_size = 720

def resize_aspect_fit():
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
             new_im.save(out_path + '/' + f + 'resized.jpg', 'JPEG', quality=90)

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
            resized_image.save(f"{out_path}/f{str(index)}_{size[0]}x{size[1]}{extension}")
            index+=1
            
#resize_aspect_fit()
resize_images([(path + '/' + x) for x in dirs], (1280,720))