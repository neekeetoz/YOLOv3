import os

# Читаем файлы
with open("input/signs_classes_720p.txt", "r", encoding='utf8') as f:
    classes = [line.strip() for line in f.readlines()]

with open("model_data/signs_annotations_720p.txt", 'r', encoding='utf8') as f:
    annotations = [line.strip() for line in f.readlines()]

# Создаем словарь с количеством экземпляров каждого класса
counts = {}
for annotation in annotations:
    boxes = annotation.split()[1:]
    for box in boxes:
        class_id = int(box.split(',')[-1])
        if class_id not in counts:
            counts[class_id] = 0
        counts[class_id] += 1

# Удаляем классы, у которых экземпляров меньше или равно 10
filtered_classes = []
filtered_counts = {}
for class_id, count in counts.items():
    if count > 10:
        filtered_classes.append(classes[class_id])
        filtered_counts[len(filtered_classes)-1] = count

# Удаляем строки с названиями удаленных классов из coco_classes.txt
with open('input/coco_classes_filtered.txt', 'w') as f:
    for class_name in filtered_classes:
        f.write(class_name + '\n')

# Обновляем class_id в аннотациях на новые значения
with open('input/annotations_filtered.txt', 'w') as f:
    for annotation in annotations:
        image_path = annotation.split()[0]
        boxes = annotation.split()[1:]
        filtered_boxes = []
        for box in boxes:
            class_id = int(box.split(',')[-1])
            if class_id in filtered_counts:
                if classes[class_id] in filtered_classes:
                    new_class_id = filtered_classes.index(classes[class_id])
                    filtered_boxes.append(','.join(box.split(',')[:-1]) + ',' + str(new_class_id))
        if len(filtered_boxes) > 0:
            f.write(image_path + ' ' + ' '.join(filtered_boxes) + '\n')
