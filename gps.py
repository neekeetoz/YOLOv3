import subprocess

with open('result.txt', 'wb', 0) as file:
    subprocess.run('exiftool -a -ee -GPS* test_2.mov', stdout=file, check=True)