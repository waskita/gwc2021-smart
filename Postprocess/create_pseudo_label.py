
# -*- coding: utf-8 -*-

import cv2
import pandas as pd
import os

csvpath = '../ScaledYOLOv4_linearhead/linearhead.csv'
imgfolder = './test'
savepath = './pseudo_label/test'
#oimgsz = 1024

os.makedirs(savepath,exist_ok=True)
test_df = pd.read_csv(csvpath)

for index, row in test_df.iterrows():
    image_name, BoxesString, domain = row['image_name'], row['PredString'], row['domain']
    #label txt
    labelfile = open(os.path.join(savepath,image_name+'.txt'),'w')
    
    #read image
    H = 1024
    W = 1024
    
    message = ''
    if BoxesString != 'no_box':
        Boxes = BoxesString.split(';')
        for box in Boxes:
            box = box.split()
            xc = (int(box[0])+int(box[2])) / 2 / W
            yc = (int(box[1])+int(box[3])) / 2 / H
            w = (int(box[2]) - int(box[0])) / W
            h = (int(box[3]) - int(box[1])) / H
            message += '0 %f %f %f %f\n'%(xc, yc, w, h)
            #cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), (0,0,255),4)
    labelfile.write(message)
    labelfile.close()
    
    
    
