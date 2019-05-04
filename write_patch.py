import pandas as pd
import numpy as np
import cv2
import os
import json
import re

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

    
    # for patches
    names=image_name(path)
    name_label=[]
    y=[]
    for name in names:
        save=name
        if 'W WHITEHALL' in name:
            continue
        name=re.findall('(.+)_',name)
        print(name)
        name=name[0]+'.jpg'
        y.append(new[name])
        name=save+'#'+str(new[name])
        name_label.append(name)
    
    # for the entire building
    #names=image_name(image_path)
    #name_label=[]
    #y=[]
    #for name in names:
    #    y.append(new[name])#'[0,1,2,...]'
    #    name=name+'#'+str(new[name]) # 'dbf.jpg#0'
    #    name_label.append(name)
    #with open('data.txt', 'w') as f:
    with open('data_patch.txt', 'w') as f:
        for item in name_label:
            f.write("%s\n" % item)
    return y,name_label
    

json_path='/home/lyuyue/Downloads/json/db_v1.json'
#path = '/home/lyuyue/Downloads/new'
path = '/home/lyuyue/Downloads/patch_select1'
image_names=image_name(path)
y,name_label=label(json_path,path,image_names)
print('a')