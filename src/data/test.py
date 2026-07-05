from PIL import Image
import os
import glob

CDIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CDIR,"../../"))             #abspath converts relative path into absolute full PATHHHHHHH
SPRITE_DIR = os.path.join(ROOT_DIR,"data","raw","sprites")
print(ROOT_DIR)
"""print(ROOT_DIR)
print(SPRITE_DIR)
print(os.listdir(os.path.dirname(__file__))) #prints contents of the directory of THIS file
print(os.listdir(SPRITE_DIR))
"""

# with Image.open(os.path.join(SPRITE_DIR,"1.png")) as im:
#     im.show()


mandel_size = (4000 , 4000)                 #in xpx,ypx
# mandel_extent = (-2.0, -1.5, 1.0, 1.5)    #x0,y0,x1,y1 on each side of the image
mandel_extent = (-1.5, -1, 0.5, 1.2)
img = Image.effect_mandelbrot(mandel_size, mandel_extent, 100)
img.show()
