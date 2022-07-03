# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


#save results to csv
def encode_boxes(boxes):
    if len(boxes) >0:
        boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
        BoxesString = ";".join(boxes)
    else:
        BoxesString = "no_box"
    return BoxesString
####
def rm_small_in_big(Boxes):
    Boxes_w = Boxes[:,2] - Boxes[:,0]
    Boxes_h = Boxes[:,3] - Boxes[:,1]
    Boxes_area = Boxes_w * Boxes_h
    
    descend_order = np.argsort(-Boxes_area)
    
    Boxes = Boxes[descend_order]
    
    
    num = 0
    selected_boxs = []
    bord = 5
    
    Boxes_copy = Boxes.copy()
    for i, box in enumerate(Boxes):
        if box in Boxes_copy:
            selected_boxs.append(box)
            left_top =  (abs(box[0] - Boxes_copy[:,0])<=bord) & (abs(box[1] - Boxes_copy[:,1])<=bord)
            left_bottom = (abs(box[0] - Boxes_copy[:,0])<=bord) & (abs(box[3] - Boxes_copy[:,3])<=bord)
            right_top = (abs(box[2] - Boxes_copy[:,2])<=bord) & (abs(box[1] - Boxes_copy[:,1])<=bord)
            right_bottom = (abs(box[2] - Boxes_copy[:,2])<=bord) & (abs(box[3] - Boxes_copy[:,3])<=bord)
            
            xx1 = np.maximum(box[0], Boxes_copy[:,0])
            yy1 = np.maximum(box[1], Boxes_copy[:,1])
            xx2 = np.minimum(box[2], Boxes_copy[:,2])
            yy2 = np.minimum(box[3], Boxes_copy[:,3])
        
            # calibrated IoU
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            Boxes_w = Boxes_copy[:,2] - Boxes_copy[:,0]
            Boxes_h = Boxes_copy[:,3] - Boxes_copy[:,1]
            Boxes_area = Boxes_w * Boxes_h
            
            inter_ratio = inter / Boxes_area
            
            select = (left_top | left_bottom | right_top | right_bottom) & (inter_ratio >=0.9)
            select_index = np.argwhere(select)#index to delete
            Boxes_copy = np.delete(Boxes_copy,select_index,axis=0)
        else:
            print("deleted!")
    
    num = num + len(Boxes)-len(selected_boxs)
    return selected_boxs, num
    
if __name__ == '__main__':
    model_csvpath = './all_ensemble.csv'

    
    model_df = pd.read_csv(model_csvpath)
       
    results = []
    num = 0
    for index, row in model_df.iterrows():
        image_name, BoxesString, domain = row['image_name'], row['PredString'], row['domain']
        #BoxesString3 = model3_df.iloc[index]['PredString']
        
        if BoxesString != 'no_box':
            Boxes = BoxesString.split(';')
            Boxeslist = []
            for box in Boxes:
                box = box.split()
                boxlist = [float(x) for x in box]
                Boxeslist.append(boxlist)        
                
            Boxes = np.array(Boxeslist)
            final_box, delete_num = rm_small_in_big(Boxes)
            num += delete_num
        else:
            final_box = ''
        PredString = encode_boxes(final_box)
                  
        #if delete_num:                
        results.append([image_name,PredString,domain])
        print(f'{image_name} record !')
    #save csv
    file_out_path = './submission.csv'
    results = pd.DataFrame(results,columns =["image_name","PredString","domain"])
    results.to_csv(file_out_path,index=False)
   # print(f'delete {num} bounding boxes')