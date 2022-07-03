# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def box_fusion(cat_predictions, img_size, iou_thresh=0.6, num=1):
    """
    args:
        cat_bboxes: concatenated boxes of origin & horizontal flip & vertical flip (format: XYWH)
        cat_scores: concatenated scores of origin & horizontal flip & vertical flip
        iou_thresh: iou threshold between two similar predctions

    output:
        final_bboxes: fused boxes
        final_scores: fused scores

    """
    cat_bboxes = cat_predictions[:,:4]
    cat_scores = cat_predictions[:,4]

    final_bboxes = []
    final_scores = []
    visit_id = set()
    for idx, boxes in enumerate(cat_bboxes):

        # pass processed box
        if idx in visit_id:
            continue

        # calculate IoU between current box and all boxes
        xx1 = np.maximum(boxes[0], cat_bboxes[:,0])
        yy1 = np.maximum(boxes[1], cat_bboxes[:,1])
        xx2 = np.minimum(boxes[2], cat_bboxes[:,2])
        yy2 = np.minimum(boxes[3], cat_bboxes[:,3])
    
        # calibrated IoU
        w = np.maximum(0.0, xx2 - xx1+1)
        h = np.maximum(0.0, yy2 - yy1+1)
        inter = w * h
        overlap = inter / ((boxes[2]-boxes[0]+1)*(boxes[3]-boxes[1]+1) + 
                (cat_bboxes[:,2]-cat_bboxes[:,0]+1)*(cat_bboxes[:,3]-cat_bboxes[:,1]+1) - inter)

        # keep box if there are two similar predictions
        if sum(overlap > iou_thresh) >= num:
            valid_ids = np.where(overlap>iou_thresh)[0]
            for valid_id in valid_ids:
                visit_id.add(int(valid_id))
            
            sel_boxes = cat_bboxes[overlap>iou_thresh]
            sel_scores = cat_scores[overlap>iou_thresh]

            weight = (sel_scores / sel_scores.sum()).reshape(-1,1)
            weight = np.concatenate((weight, weight, weight, weight), axis=1)

            weighted_boxes = (weight * sel_boxes).sum(axis=0)
            final_bboxes.append(weighted_boxes)
            final_scores.append(sel_scores.mean())
        
    final_bboxes, final_scores = np.array(final_bboxes), np.array(final_scores)
    return final_bboxes, final_scores

#save results to csv
def encode_boxes(boxes):
    if len(boxes) >0:
        boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
        BoxesString = ";".join(boxes)
    else:
        BoxesString = "no_box"
    return BoxesString
####
if __name__ == '__main__':
    model1_csvpath = 'baseline_ensemble.csv'
    model2_csvpath = 'linearhead_ensemble.csv'
    model3_csvpath = '../ScaledYOLOv4/inference/output/baseline.csv'
    
    model1_df = pd.read_csv(model1_csvpath)
    model2_df = pd.read_csv(model2_csvpath)
    model3_df = pd.read_csv(model3_csvpath)
       
    results = []
    for index, row in model1_df.iterrows():
        mixBoxes = []
        image_name, BoxesString1, domain = row['image_name'], row['PredString'], row['domain']
        BoxesString2 = model2_df.iloc[index]['PredString']
        BoxesString3 = model3_df.iloc[index]['PredString']
        
        if BoxesString1 != 'no_box':
            Boxes1 = BoxesString1.split(';')
            for box in Boxes1:
                box = box.split()
                boxfloat = [float(x) for x in box]
                mixBoxes.append(boxfloat)
        
        if BoxesString2 != 'no_box':
            Boxes2 = BoxesString2.split(';')
            for box in Boxes2:
                box = box.split()
                boxfloat = [float(x) for x in box]
                mixBoxes.append(boxfloat)
        
        if BoxesString3 != 'no_box':
            Boxes3 = BoxesString3.split(';')
            for box in Boxes3:
                box = box.split()
                boxfloat = [float(x) for x in box]
                mixBoxes.append(boxfloat)
            
        mixBoxes = np.asarray(mixBoxes)
        if len(mixBoxes) != 0:
            final_box, _ = box_fusion(mixBoxes, 1024, iou_thresh=0.6, num=2)
        else:
            final_box = ''
        PredString = encode_boxes(final_box)
                                  
        results.append([image_name,PredString,domain])
        print(f'{image_name} record !')
    #save csv
    file_out_path = 'all_ensemble.csv'
    results = pd.DataFrame(results,columns =["image_name","PredString","domain"])
    results.to_csv(file_out_path,index=False)