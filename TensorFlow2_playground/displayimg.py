from PIL import Image
import numpy as np

def dispalyImg(width,height,str: name):
    w,h = 28, 28
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:w/2, 0:h/2] = [255, 0, 0] # red patch in upper left
    img = Image.fromarray(data, 'RGB')
    return img.show() and img.save(f'{name}.png')

