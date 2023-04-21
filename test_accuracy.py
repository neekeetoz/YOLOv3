import math

def get_obj_coord(latitude, longitude, width, sideImage, image_width):
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
    #distance_forward *= 6 if sideImage == SideImage.RIGHT else -6
    # фокусное расстояние
    f = 1000
    
    angle = math.radians(170)
    k = 2 * math.tan(angle/2) / image_width
    # расстояние в сторону от камеры в метрах
    distance_to_side = k * f / (1 - k * distance_forward)
    
    if isNorth:
        longitude -= distance_to_side / 1000 / multiple_longitude
    if isSouth:
        longitude += distance_to_side / 1000 / multiple_longitude
    if isEast:
        latitude += distance_forward / 1000 / multiple_latitude
    if isWest:
        latitude -= distance_forward / 1000 / multiple_latitude
    return (round(latitude, 6), round(longitude, 6))

# часть изображения
class SideImage:
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    
# TEST
last_latitude = 0
last_longitude = 0
#get_obj_coord(latitude, longitude, width, sideImage, image_width)