#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
import pandas as pd
import numpy as np
import cv2
import os
import json

def image_name(path):
    img_name=[]
    files=os.listdir(path)
    for name in files:
        if name=='.DS_Store'or name=='_DS_Store':
            continue
        img_name.append(name)
    return img_name

def label(json_path,image_path,img_name):
    new={}
    with open(json_path, 'r') as f:
        datastore = json.load(f)
    for db in datastore:
        src=db['img_src'][29:]
        src = src.replace('_',' ')
        #house=src[29:]
        if db['quality']=='H':
            q=0
        elif db['quality']=='M':
            q=1
        else:
            q=2
        new[src]=q
    names=image_name(image_path)
    name_label=[]
    y=[]
    for name in names:
        y.append(new[name])#'[0,1,2,...]'
        name=name+'#'+str(new[name]) # 'dbf.jpg#0'
        name_label.append(name)
    with open('data.txt', 'w') as f:
        for item in name_label:
            f.write("%s\n" % item)
    return y,name_label
    

json_path='/home/lyuyue/Downloads/json/db_v1.json'
path = '/home/lyuyue/Downloads/new'
image_names=image_name(path)
y,name_label=label(json_path,path,image_names)
print('a')
