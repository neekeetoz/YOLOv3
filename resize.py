from PIL import Image
import os, sys

path = "C:/Users/M.Kalentyiev/Downloads/images/картинки - Саранск/2"
out_path = "C:/Users/M.Kalentyiev/Downloads/images/416-картинки - Саранск/2"
if not os.path.exists(out_path):
    os.makedirs(out_path)
dirs = os.listdir( path )

final_size = 416

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
resize_aspect_fit()