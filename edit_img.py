from PIL import Image, ImageFilter, ImageEnhance
import os

def edit_img(path, name, file_annotation, annotation):
    
    image = Image.open(path + name)
    n, _ = os.path.splitext(name)
    # размытие
    img_out = image.filter((ImageFilter.BoxBlur(1)))
    img_out.save(path + n + '1.jpg')
    file_annotation.write(path + n + '1.jpg ' + annotation)

    # яркость
    img_out = ImageEnhance.Contrast(image).enhance(0.5)
    img_out.save(path + n + '2.jpg')
    file_annotation.write(path + n + '2.jpg ' + annotation)
    img_out = ImageEnhance.Contrast(image).enhance(1.5)
    img_out.save(path + n + '3.jpg')
    file_annotation.write(path + n + '3.jpg ' + annotation)

def get_names(path):
    return '\n'.join(os.listdir(path))

#main
path = 'images/'
with open('input/train_27c.txt', 'r+') as f:
    # f.write(get_names(path))
    images = os.listdir(path)
    lines = f.readlines()
    for image in images:
        for s in lines:
            annotation = s.split(' ', 1)
            if annotation[0] == (path + image):
                break
        if len(annotation) > 1:
            edit_img(path, image, f, annotation[1])
        else:
            edit_img(path, image, f, '\n')
        print(image)
print('end')