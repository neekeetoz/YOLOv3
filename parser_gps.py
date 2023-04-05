import subprocess

class GPS(object):
    def __init__(self):
        self.date = None
        self.time = None
        self.latitude = None
        self.longitude = None
        self.speed = None

    def __str__(self):
        return self.latitude + ' ' + self.longitude

# сохраняет gps данные в указ
def get_gps_data_from_file(path):
    with open('result.txt', 'wb', 0) as file:
        # subprocess.run(f'exiftool -a -ee -GPS* {path}', stdout=file, check=True)
        subprocess.run(f'exiftool -a -ee -p "Date/Time : $gpsdatetime\nLatitude : $gpslatitude\nLongitude : $gpslongitude\nSpeed : $gpsspeed" {path}', stdout=file, check=True)

    # считываем gps
    gps = []
    with open('result.txt', 'r') as file:
        lines = file.readlines()
        i = 0
        gps.append(GPS())
        for line in lines:
            f = line.rstrip().split(' :')
            if "Date/Time" == f[0]:
                gps[i].date = '/'.join(f[1].split(' ')[1].split(':')[::-1])
                gps[i].time = f[1].split(' ')[2].split('.')[0]
            elif "Latitude" == f[0]:
                gps[i].latitude = convert_to_decimal_coord(convert_coord_to_int(f[1]))
            elif "Longitude" == f[0]:
                gps[i].longitude = convert_to_decimal_coord(convert_coord_to_int(f[1]))
            elif "Speed" == f[0]:
                gps[i].speed = f[1]
                i += 1
                gps.append(GPS())
    return gps

# преобразует координаты из строкового представления в числовое
def convert_coord_to_int(coord):
    l = len(coord)
    integ = []
    i = 0
    while i < l:
        s_int = ''
        a = coord[i]
        while '0' <= a <= '9':
            s_int += a
            i += 1
            if i < l:
                a = coord[i]
            else:
                break
        i += 1
        if s_int != '':
            integ.append(int(s_int))
    return integ

# возвращает координаты в десятичном виде
def convert_to_decimal_coord(integ):
    return round((integ[0] + integ[1] / 60 + float(str(integ[2]) + '.' + str(integ[3])) / 3600), 6)

    

    

