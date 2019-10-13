import os
from PIL import Image
import sys

#获取path目录下的所有文件
#def get_imlist(path):
    #return[os.path.join(path,f) for f in os.listdir(path)]

#def change_size(path):
    #directorys=get_imlist(path)
i = 0
path = "E://海报拼接//素材//"
dir = [os.path.join(path, f) for f in os.listdir(path)]
for directory in dir:
    try:
        img=Image.open(directory)
        to_save="E://海报拼接//resized素材//" + str(i) + '.jpg'
        new_width=120
        new_height=120
        out=img.resize((new_width,new_height),Image.ANTIALIAS)
        out.save(to_save)
        #os.remove(directory)
        i += 1
        print('{} completed'.format(i))
    except:
        print('fail')

#change_size("E://海报拼接//素材//")
#change_size("/home/winney/image_db/bmp")
#change_size("/home/winney/image_db/png")
# E://ScrollBarTest//ScrollBarTest//bin//Debug//Water lilies.jpg
