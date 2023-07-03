import numpy as np
import os
import PIL
import PIL.Image
#import tensorflow as tf
import xml.etree.ElementTree as ET

def parse(file):
    tree = ET.parse(file)
    root = tree.getroot()

    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        objects.append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })

    return {
        filename: {
            'width': width,
            'height': height,
            'objects': objects
        }
    }

annoDir = 'data/annotations/Annotation'
imgDir = 'data/images/Images'
folders = os.listdir(annoDir)

annos = {}
for i in folders:
    files = os.listdir(annoDir + '/' + i)
    annos.update(parse(annoDir + '/' + i + '/' + files))
