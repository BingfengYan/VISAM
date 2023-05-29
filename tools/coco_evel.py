import os
import argparse
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
parser.add_argument('--det_root', default='tracker', type=str)
args = parser.parse_args()

cocoGt = COCO(annotation_file='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/annotations/val.json')


det_root = args.det_root
tracklets = defaultdict()

detRes = []
for img_id in cocoGt.getImgIds():
    img = cocoGt.loadImgs(img_id)
    
    vid_name = img[0]['file_name'][:14]
    frame_id = img[0]['frame_id'] 
    
    if vid_name not in tracklets:
        tracklets[vid_name] = defaultdict(list)
        for line in open(os.path.join(det_root, vid_name+'.txt')):
            t, id, *xywhs = line.split(',')[:7]
            t, id = map(int, (t, id))
            tracklets[vid_name][t].append((id, *map(float, xywhs)))
    
    labels = tracklets[vid_name][frame_id]
    
    for l in labels:
        ann = defaultdict()
        ann['image_id'] = img[0]['id'] 
        ann['bbox'] = list(l[1:5])
        ann['category_id'] = 1
        ann['score'] = l[5]
        detRes.append(ann)

cocoDt = cocoGt.loadRes(detRes)  #自己的生成的结果的路径及文件名，json文件形式
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()