import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from typing import final

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadHSVImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import pandas as pd
import numpy as np


def box_fusion(cat_predictions, img_size, iou_thresh=0.6, num=2):
    """
    args:
        cat_bboxes: concatenated boxes of origin & horizontal flip & vertical flip (format: XYWH)
        cat_scores: concatenated scores of origin & horizontal flip & vertical flip
        iou_thresh: iou threshold between two similar predctions

    output:
        final_bboxes: fused boxes
        final_scores: fused scores

    """
    cat_bboxes = np.clip(cat_predictions[:,:4].cpu().numpy(), a_min=0, a_max=img_size)
    cat_scores = cat_predictions[:,4].cpu().numpy()

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
    
@torch.no_grad()
def detect(save_img=False):
    readpath = "./sample_submission.csv"

    out, source, weights, view_img, save_txt, imgsz, fusion_num = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.fusion_num
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    os.makedirs(out,exist_ok=True)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    
    #read csv
    test_df = pd.read_csv(readpath)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    #results for submission csv
    results = []
    #
    def encode_boxes(boxes):
        if len(boxes) >0:
            boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
            BoxesString = ";".join(boxes)
        else:
            BoxesString = "no_box"
        return BoxesString
    ####
    paths = []
    imgs = []
    im0ss = []
    results = []

    a = 0.5
    for path, img, im0s, vid_cap in dataset:
        paths.append(path)
        #imgs.append(np.clip((img*a),0,255).astype(np.uint8))
        imgs.append(img)
        im0ss.append(im0s)

    for index, row in test_df.iterrows():
        image_name, PredString, domain = row['image_name'], row['PredString'], row['domain']

        index = paths.index(source+image_name+'.png')
        path = paths[index]
        img = imgs[index]
        im0s = im0ss[index]

        #*********upside-down*************
        #upside-down flip
        imgud = np.flip(img,1).copy()
        imgud = torch.from_numpy(imgud).to(device)
        imgud = imgud.half() if half else imgud.float()  # uint8 to fp16/32
        imgud /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imgud.ndimension() == 3:
            imgud = imgud.unsqueeze(0)

        predud = model(imgud, augment=opt.augment)[0]
        predud = non_max_suppression(predud, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        detud = predud[0]
        #convert to origin box 
        if detud is not None:
            temp = detud[:,3].clone()
            detud[:,3] = imgsz  - detud[:,1]
            detud[:,1] = imgsz  - temp
        #*********left-right*************
        #left-right flip
        imglr = np.flip(img,2).copy()
        imglr = torch.from_numpy(imglr).to(device)
        imglr = imglr.half() if half else imglr.float()  # uint8 to fp16/32
        imglr /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imglr.ndimension() == 3:
            imglr = imglr.unsqueeze(0)

        predlr = model(imglr, augment=opt.augment)[0]
        predlr = non_max_suppression(predlr, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        detlr = predlr[0]
        #convert to origin box 
        if detlr is not None:
            temp = detlr[:,2].clone()
            detlr[:,2] = imgsz  - detlr[:,0]
            detlr[:,0] = imgsz  - temp
        #***********rotate+90****************
        def flip90_right(arr):
            new_arr = arr.transpose(0,2,1)
            new_arr = new_arr[:,:,::-1]

            return new_arr
        
        imgR90 = flip90_right(img).copy()
        imgR90 = torch.from_numpy(imgR90).to(device)
        imgR90 = imgR90.half() if half else imgR90.float()  # uint8 to fp16/32
        imgR90 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imgR90.ndimension() == 3:
            imgR90 = imgR90.unsqueeze(0)

        predR90 = model(imgR90, augment=opt.augment)[0]
        predR90 = non_max_suppression(predR90, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        detR90 = predR90[0]
        #convert to origin box 
        if detR90 is not None:
            tempdetR90 = detR90.clone()
            detR90[:,0] = tempdetR90[:,1]
            detR90[:,1] = imgsz - tempdetR90[:,2]
            detR90[:,2] = tempdetR90[:,3]
            detR90[:,3] = imgsz - tempdetR90[:,0]
        #***********rotate-90****************
        def flip90_left(arr):
            new_arr = arr.transpose(0,2,1)
            new_arr = new_arr[:,::-1,:]

            return new_arr

        imgL90 = flip90_left(img).copy()
        imgL90 = torch.from_numpy(imgL90).to(device)
        imgL90 = imgL90.half() if half else imgL90.float()  # uint8 to fp16/32
        imgL90 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imgL90.ndimension() == 3:
            imgL90 = imgL90.unsqueeze(0)

        predL90 = model(imgL90, augment=opt.augment)[0]
        predL90 = non_max_suppression(predL90, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        detL90 = predL90[0]
        #convert to origin box 
        if detL90 is not None:
            tempdetL90 = detL90.clone()
            detL90[:,0] = imgsz - tempdetL90[:,3]
            detL90[:,1] = tempdetL90[:,0]
            detL90[:,2] = imgsz - tempdetL90[:,1]
            detL90[:,3] = tempdetL90[:,2]
        #*********origin*************
        #origin keep
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=opt.augment)[0]#Box (center x, center y, width, height)
        # Apply NMS
        ## Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # Process detections
        det = pred[0]

        #************concatenate predicted boxes**********************
        detlist = [det, detlr, detud, detR90, detL90, #detR180
        ] 
        None_num = 0
        for d in detlist:
            if d is None:
                None_num += 1
        if None_num != len(detlist):
            det = torch.cat([x for x in detlist if x is not None])
            #**************box fusion****************
            final_boxes, final_scores = box_fusion(det, img_size=imgsz, num=fusion_num)
        else:
            final_boxes = np.empty(0)
            final_scores = np.empty(0)

        if final_boxes.shape[0] != final_scores.size:
            c = 1
        if final_boxes.size !=0:
            #***********Non-maximum supression again**************
            #(x1,y1,x2,y2) to (center x, center y, width, height)
            box_cx  = (final_boxes[:,0] + final_boxes[:,2]) / 2
            box_cy  = (final_boxes[:,1] + final_boxes[:,3]) / 2
            box_w = final_boxes[:,2] - final_boxes[:,0]
            box_h = final_boxes[:,3] - final_boxes[:,1]
            final_boxes[:,0] = box_cx
            final_boxes[:,1] = box_cy
            final_boxes[:,2] = box_w
            final_boxes[:,3] = box_h

            final_boxes = np.hstack((final_boxes, final_scores.reshape(-1,1))) 
            cls_scores =  np.ones((final_boxes.shape[0],1),dtype=np.float)
            final_boxes = np.hstack((final_boxes, cls_scores))
            final_pred = torch.from_numpy(final_boxes).to(device).unsqueeze(0)

            # Apply NMS
            final_pred = non_max_suppression(final_pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            final_det = final_pred[0]
            final_boxes = final_det[:,:4].cpu().numpy()

        p, s, im0 = path, '', im0s
        save_path = str(Path(out) / Path(p).name)
        txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # Rescale boxes from img_size to im0 size
        if None_num==len(detlist) or final_boxes.size==0:
            PredString = "no_box"
        else:
            final_boxes = scale_coords(img.shape[2:], torch.from_numpy(final_boxes), im0.shape).round()
            final_boxes = final_boxes[((final_boxes[:,2]-final_boxes[:,0])>opt.side_thres) & ((final_boxes[:,3]-final_boxes[:,1])>opt.side_thres)]
            final_boxes_w = final_boxes[:,2]-final_boxes[:,0]
            final_boxes_h = final_boxes[:,3]-final_boxes[:,1]
            final_boxes_area = final_boxes_w.mul(final_boxes_h)
            final_boxes = final_boxes[final_boxes_area > opt.area_thres]

            PredString = encode_boxes(final_boxes.cpu().numpy())#det x1,y1,x2,y2,conf


        results.append([image_name,PredString,domain])
        print(f'{image_name} record !')
    
    #save csv
    file_out_path = str(Path(out) / opt.output_filename)
    results = pd.DataFrame(results,columns =["image_name","PredString","domain"])
    results.to_csv(file_out_path,index=False)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--fusion-num', type=int, default=2)
    parser.add_argument('--area-thres', type=float, default=0, help='object area threshold')
    parser.add_argument('--side-thres', type=float, default=20, help='object area threshold')
    parser.add_argument('--output-filename', type=str, default='submission.csv', help='output filename')
    opt = parser.parse_args()

    opt.source = "./test/"
    opt.conf_thres = 0.4
    opt.img_size = 1024
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
