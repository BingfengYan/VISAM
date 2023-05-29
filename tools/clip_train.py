import os
import cv2
import numpy as np
from collections import defaultdict


root_data = '/home/hadoop-vacv/yanfeng/data/MOT/MOT17_all/train'
vids = os.listdir(root_data)

for v in vids:
    if 'SDP' in v:
        labels_full = defaultdict(list)
        gt_path = os.path.join(root_data, v, 'gt', 'gt.txt')
        for l in open(gt_path):
            t, i, *xywh = l.strip().split(',')
            labels_full[int(t)].append([i, *xywh])
        imgs_root = os.path.join(root_data, v, 'img1')
        imgs_path = sorted(os.listdir(imgs_root))
        
        for ith, img_p in enumerate(imgs_path):
            if ith < (len(imgs_path)+1)//2:
                save_img = os.path.join(imgs_root, img_p).replace('MOT17_all', 'MOT17')
                save_label = os.path.join(root_data, v, 'gt', 'gt.txt').replace('MOT17_all', 'MOT17')
                print('train: %d', save_img)
            else:
                save_img = os.path.join(imgs_root, img_p).replace('MOT17_all', 'MOT17').replace('train', 'val')
                save_label = os.path.join(root_data, v, 'gt', 'gt.txt').replace('MOT17_all', 'MOT17').replace('train', 'val')
                print('val: %d', save_img)
            os.makedirs(os.path.dirname(save_label), exist_ok=True)
            with open(save_label, 'a+') as f:
                if ith+1 in labels_full:
                    for l in labels_full[ith+1]:
                        f.write('%d,%s,%s,%s,%s,%s,%s,%s,%s\n'%(ith+1, *l))
            img = cv2.imread(os.path.join(imgs_root, img_p))
            os.makedirs(os.path.dirname(save_img), exist_ok=True)
            cv2.imwrite(save_img, img)