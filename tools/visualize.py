# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


from collections import defaultdict
from glob import glob
import json
import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment as linear_assignment

# 计算两个box的IOU
def bboxes_iou(bboxes1,bboxes2):
	bboxes1 = np.transpose(bboxes1)
	bboxes2 = np.transpose(bboxes2)

	# 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
	int_ymin = np.maximum(bboxes1[0][:, None], bboxes2[0])
	int_xmin = np.maximum(bboxes1[1][:, None], bboxes2[1])
	int_ymax = np.minimum(bboxes1[2][:, None], bboxes2[2])
	int_xmax = np.minimum(bboxes1[3][:, None], bboxes2[3])

	# 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
	int_h = np.maximum(int_ymax-int_ymin,0.)
	int_w = np.maximum(int_xmax-int_xmin,0.)

	# 计算IOU
	int_vol = int_h * int_w # 交集面积
	vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) # bboxes1面积
	vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1]) # bboxes2面积
	IOU = int_vol / (vol1[:, None] + vol2 - int_vol) # IOU=交集/并集
	return IOU

def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]

with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/det_db_motrv2.json") as f:
    det_db = json.load(f)

def process(trk_path, img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape
    command = [
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/detrex/bin/ffmpeg",
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{w}x{h}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', '20',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-s', f'{w//2*2}x{h//2*2}',
        '-an',  # Tells FFMPEG not to expect any audio
        '-loglevel', 'error',
        # '-crf', '26',
        '-b:v', '0',
        '-pix_fmt', 'yuv420p'
    ]
    # writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)
    fps = 16 
    size = (w,h) 
    videowriter = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)


    tracklets = defaultdict(list)
    for line in open(trk_path):
        t, id, *xywhs = line.split(',')[:7]
        t, id = map(int, (t, id))
        x, y, w, h, s = map(float, xywhs)
        tracklets[t].append((id, *map(int, (x, y, x+w, y+h))))

    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        det_bboxes = []
        motr_bboxes = []
        for det in det_db[path.replace('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/', '').replace('.jpg', '.txt').replace('dancetrack/', 'DanceTrack/')]:
            x1, y1, w, h, s = map(float, det.strip().split(','))
            x1, y1, w, h = map(int, [x1, y1, w, h])
            im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 2)
            im = cv2.putText(im, '%0.2f'%s, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            det_bboxes.append([x1, y1, x1+w, y1+h])
        for j, x1, y1, x2, y2 in tracklets[i + 1]:
            im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 2)
            im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(j), 1)
            motr_bboxes.append([x1, y1, x2, y2])
        
        det_bboxes = np.array(det_bboxes)
        motr_bboxes = np.array(motr_bboxes)
        ious = bboxes_iou(det_bboxes, motr_bboxes)
        matching = linear_assignment(-ious)
        matched = sum(ious[matching[0], matching[1]] > 0.5)
        im = cv2.putText(im, f"{matched}/{len(det_bboxes)}/{len(motr_bboxes)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, get_color(j), 3)
        cv2.putText(im, "{}".format(os.path.basename(path)[:-4]), (120,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
        # writing_process.stdin.write(im.tobytes())
        videowriter.write(im)
        
    videowriter.release()


def process_compare(trk_path1, trk_path2, gt_path, img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape
    command = [
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/detrex/bin/ffmpeg",
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{w}x{h}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', '20',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-s', f'{w//2*2}x{h//2*2}',
        '-an',  # Tells FFMPEG not to expect any audio
        '-loglevel', 'error',
        # '-crf', '26',
        '-b:v', '0',
        '-pix_fmt', 'yuv420p'
    ]
    # writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)
    fps = 16 
    size = (w,h*2) 
    videowriter = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)


    tracklets = defaultdict(list)
    for line in open(trk_path1):
        t, id, *xywhs = line.split(',')[:7]
        t, id = map(int, (t, id))
        x, y, w, h, s = map(float, xywhs)
        tracklets[t].append((id, *map(float, (x, y, x+w, y+h, s))))
    tracklets2 = defaultdict(list)
    for line in open(trk_path2):
        t, id, *xywhs = line.split(',')[:7]
        t, id = map(int, (t, id))
        x, y, w, h, s = map(float, xywhs)
        tracklets2[t].append((id, *map(float, (x, y, x+w, y+h, s))))
    
    gtlets = defaultdict(list)
    for line in open(gt_path):
        t, id, *xywhs = line.split(',')[:7]
        t, id = map(int, (t, id))
        x, y, w, h, s = map(float, xywhs)
        gtlets[t].append((id, *map(int, (x, y, x+w, y+h))))

    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        im2 = im.copy()
        gt_bboxes = []
        motr_bboxes, motr_bboxes2 = [], []
        for j, x1, y1, x2, y2 in gtlets[i + 1]:
            w, h = x2-x1, y2-y1
            im = cv2.rectangle(im, (x1, y1), (x2, y2), (255,255,255), 2)
            im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 2)
            im = cv2.putText(im, '%0.2f'%s, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            im2 = cv2.rectangle(im2, (x1, y1), (x1+w, y1+h), (255, 255, 255), 2)
            im2 = cv2.putText(im2, '%0.2f'%s, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            gt_bboxes.append([x1, y1, x2, y2])
        for j, x1, y1, x2, y2, s in tracklets[i + 1]:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 2)
            im = cv2.putText(im, "%d/%.2f"%(j, s), (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(j), 1)
            motr_bboxes.append([x1, y1, x2, y2])
        for j, x1, y1, x2, y2, s in tracklets2[i + 1]:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            im2 = cv2.rectangle(im2, (x1, y1), (x2, y2), get_color(j), 2)
            im2 = cv2.putText(im2, "%d/%.2f"%(j, s), (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(j), 1)
            motr_bboxes2.append([x1, y1, x2, y2])
            
        gt_bboxes = np.array(gt_bboxes)
        motr_bboxes = np.array(motr_bboxes)
        ious = bboxes_iou(gt_bboxes, motr_bboxes)
        matching = linear_assignment(-ious)
        matched = sum(ious[matching[0], matching[1]] > 0.5)
        cv2.putText(im, "{}".format(os.path.basename(path)[:-4]), (120,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
        
        motr_bboxes2 = np.array(motr_bboxes2)
        ious = bboxes_iou(gt_bboxes, motr_bboxes2)
        matching = linear_assignment(-ious)
        matched2 = sum(ious[matching[0], matching[1]] > 0.5)
        if matched2 == matched:
            im = cv2.putText(im, f"{matched}/{len(gt_bboxes)}/{len(motr_bboxes)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, get_color(j), 3)
            im2 = cv2.putText(im2, f"{matched2}/{len(gt_bboxes)}/{len(motr_bboxes2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, get_color(j), 3)
        else:
            im = cv2.putText(im, f"{matched}/{len(gt_bboxes)}/{len(motr_bboxes)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            im2 = cv2.putText(im2, f"{matched2}/{len(gt_bboxes)}/{len(motr_bboxes2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(im2, "{}".format(os.path.basename(path)[:-4]), (120,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
        
        im = np.concatenate([im, im2], axis=0)
        # writing_process.stdin.write(im.tobytes())
        videowriter.write(im)
        
    videowriter.release()


if __name__ == '__main__':
    jobs = os.listdir("exps/motrv2_group/run2/tracker_max_max/")
    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    jobs = sorted(jobs)[rank::ws]
    for seq in jobs:
        seq = 'dancetrack0004.txt'
        print(seq)
        trk_path = "exps/motrv2_group/run2/tracker_max_max/" + seq
        gt_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val/%s/gt/gt.txt"%(seq[:-4])
        
        # trk_path = "/data/Dataset/mot/DancdancetrackeTrack/val/dancetrack0010/gt/gt.txt"

        img_list = glob(f"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val/{seq[:-4]}/img1/*.jpg")
        # process(trk_path, img_list, f'tmp/{seq[:-4]}.avi')
        
        trk_path2 = "exps/motrv2_group/run2/tracker_max_min/" + seq
        process_compare(trk_path, trk_path2, gt_path, img_list, f'tmp/{seq[:-4]}.avi')
        break
