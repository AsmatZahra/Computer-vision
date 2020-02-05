from PIL import Image
import os, sys
# path = "D:/NUST/PhD/Computer Vision/Assignment 2/INRIA_Dataset_Samples_shortt/Test/pos/"
path = "D:/NUST/PhD/Computer Vision/Assignment 2/A2_task1_images/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((500,500), Image.ANTIALIAS)
            imResize.save(f+'.jpg', quality=80)
            # imResize.save(f + '.jpg', quality=80)

resize()