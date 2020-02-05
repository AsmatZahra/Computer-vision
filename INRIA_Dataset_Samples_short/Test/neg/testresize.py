from PIL import Image
import glob, os

size = 160, 90

for infile in glob.glob("*.PNG"):
    # print (glob.glob("*.png"))
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size)
    im.save(file + ".thumbnail" , "PNG")
# print ("next")
for infile in glob.glob("*.JPG"):
    # print (infile)
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size)
    im.save(file+".thumbnail", "JPEG")
