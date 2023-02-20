import subprocess

class GPS(object):
    def __init__(self, date, time, latitude, longitude, speed):
        self.date = date
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.speed = speed

    def __str__(self):
        return self.latitude + ' ' + self.longitude

def get_gps_data_from_file(path):
    with open('result.txt', 'wb', 0) as file:
        subprocess.run(f'exiftool -a -ee -GPS* {path}', stdout=file, check=True)

    gps = []

    def chisla(coord):
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

    def converttodecimalcoord(integ):
        return integ[0] + integ[1] / 60 + float(str(integ[2]) + '.' + str(integ[3])) / 3600


    with open('result.txt', 'r') as file:
        lines = file.readlines()
        i = 0
        gps.append(GPS)
        for line in lines:
            f = line.rstrip().split(' :')
            if "Date/Time" == f[0].split(' ')[1]:
                gps[i].date = '/'.join(f[1].split(' ')[1].split(':')[::-1])
                gps[i].time = f[1].split(' ')[2]
            elif "Latitude" == f[0].split(' ')[1]:
                gps[i].latitude = converttodecimalcoord(chisla(f[1]))
            elif "Longitude" == f[0].split(' ')[1]:
                gps[i].longitude = converttodecimalcoord(chisla(f[1]))
            elif "Speed" == f[0].split(' ')[1]:
                gps[i].speed = f[1]
                i += 1
                gps.append(GPS)

    # for odin in gps:
    #     print(f'date: {odin.date}\ntime: {odin.time}\nlatitude: {odin.latitude}\nlongitude: {odin.longitude}\nspeed:{odin.speed}')

    return gps

