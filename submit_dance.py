# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json
import threading
import os
import random
import numpy as np
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('thirdparty/segment-anything')
from segment_anything import build_sam, SamPredictor 


np.random.seed(2020)
    
COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, lable_offset=0):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30 + lable_offset), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


'''
deep sort 中的画图方法，在原图上进行作画
'''
def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False, lable_offset=0):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score, lable_offset=lable_offset)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        if len(self.det_db):
            for line in self.det_db[f_path[:-4].replace('dancetrack/', 'DanceTrack/') + '.txt']:
                l, t, w, h, s = list(map(float, line.split(',')))
                proposals.append([(l + w / 2) / im_w,
                                    (t + h / 2) / im_h,
                                    w / im_w,
                                    h / im_h,
                                    s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5), f_path

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):  # 加载图像和proposal。并对图像颜色通道转换+resize+normalize+to_tensor。
        img, proposals, f_path = self.load_img_from_file(self.img_list[index])
        img, ori_img, proposals = self.init_img(img, proposals)
        return img, ori_img, proposals, f_path


class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

        self.sam_predictor = SamPredictor(build_sam(checkpoint="thirdparty/segment-anything/sam_vit_h_4b8939.pth"))
        _ = self.sam_predictor.model.to(device='cuda')
        
        self.save_path = 'tmp'

    @staticmethod
    def filter_dt_by_score(dt_instances, prob_threshold):
        keep = dt_instances.scores > prob_threshold
        # if keep.sum() % 5 != 0:
        #     print(dt_instances.scores)
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances, area_threshold):
        wh = dt_instances.boxes[..., 2:4] - dt_instances.boxes[..., 0:2]
        areas = wh[..., 0] * wh[..., 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances, ref_pts=None, gt_boxes=None, obj_instances=None):
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes)
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        if obj_instances is not None:
            img_show = draw_bboxes(img, np.concatenate([obj_instances.boxes, obj_instances.scores_obj.reshape(-1, 1)], axis=-1), lable_offset=-50)
        cv2.imwrite(img_path, img_show)


    def detect(self, prob_threshold=0.6, area_threshold=100, vis=False):

        track_instances = None
        det_db = []
        if len(self.args.det_db):
            with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
                det_db = json.load(f)
        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2)
        lines = defaultdict(list)
        total_dts = defaultdict(int)
        total_occlusion_dts = defaultdict(int)
        print('g_size: %d'%self.args.g_size)
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img, proposals, f_path = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

			if i == 0:
			    fps = 25
		        size = ori_img.to(torch.device('cpu')).numpy().shape[:2:-1]
		        self.videowriter = cv2.VideoWriter('visam.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)


            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                # track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            # 内部包含backboe+encode+decode+跟踪匹配关系+跟踪目标过滤（从query中过滤）
            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances_all = deepcopy(track_instances).get_bn(0)

            # filter det instances by score.
            dt_instances_all = self.filter_dt_by_score(dt_instances_all, prob_threshold)  # 保留置信度比较高的目标（因为motr内部可能会保留相对置信度高一些的目标，但输出需要输出比较高一些）
            dt_instances_all = self.filter_dt_by_area(dt_instances_all, area_threshold) # 过滤小目标
            
            active_indx = []
            full_indx = torch.arange(len(dt_instances_all), device=dt_instances_all.scores.device)
            for id in torch.unique(dt_instances_all.obj_idxes):
                indx = torch.where(dt_instances_all.obj_idxes == id)[0]
                active_indx.append(full_indx[indx][dt_instances_all.scores[indx].argmax()])
            if len(active_indx):
                active_indx = torch.stack(active_indx)
                dt_instances_all = dt_instances_all[active_indx]
            
            for g_id in range(1):
                # dt_instances = dt_instances_all[dt_instances_all.group_ids==g_id]
                dt_instances = dt_instances_all
                
                total_dts[g_id] += len(dt_instances)

                bbox_xyxy = dt_instances.boxes.tolist()
                identities = dt_instances.obj_idxes.tolist()


	            masks_all = []
	            self.sam_predictor.set_image(ori_img.to(torch.device('cpu')).numpy().copy())
            
	            for bbox, id in zip(np.array(bbox_xyxy), identities):
	                masks, iou_predictions, low_res_masks = self.sam_predictor.predict(box=bbox)
	                index_max = iou_predictions.argsort()[0]
	                masks = np.concatenate([masks[index_max:(index_max+1)], masks[index_max:(index_max+1)], masks[index_max:(index_max+1)]], axis=0)
	                masks = masks.astype(np.int32)*np.array(colors(id))[:, None, None]
	                masks_all.append(masks)
            
	            self.sam_predictor.reset_image()
	            if len(masks_all):
	                masks_sum = masks_all[0].copy()
	                for m in masks_all[1:]:
	                    masks_sum += m
	            else:
	                masks_sum = np.zeros_like(img).transpose(2, 0, 1)

	            img = ori_img.to(torch.device('cpu')).numpy().copy()[..., ::-1]
	            img = (img * 0.5 + (masks_sum.transpose(1,2,0) * 30) %128).astype(np.uint8)
	            for bbox in bbox_xyxy:
	                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), thickness=3)
	            self.videowriter.write(img)

                if vis:
                    # for visual
                    cur_vis_img_path = os.path.join(self.save_path, '%08d.jpg'%i)
                    gt_boxes = None
                    all_ref_pts = None # tensor_to_numpy(res['ref_pts'][0, :, :2])
                    self.visualize_img_with_bbox(cur_vis_img_path, ori_img.to(torch.device('cpu')).numpy().copy(), dt_instances.to(torch.device('cpu')), ref_pts=all_ref_pts, gt_boxes=gt_boxes)
                    if 'track_instances_ori' in res:
                        active_track_instances = res['track_instances_ori']
                        active_track_instances = active_track_instances[active_track_instances.scores_obj >= 0.3]
                        active_track_instances.scores = active_track_instances.scores_obj
                        active_track_instances = active_track_instances.to(torch.device('cpu'))
                        cur_vis_img_path = os.path.join(self.save_path, 'det_%08d.jpg'%i)
                        self.visualize_img_with_bbox(cur_vis_img_path, ori_img.to(torch.device('cpu')).numpy().copy(), active_track_instances, ref_pts=all_ref_pts, gt_boxes=gt_boxes)
                    
                save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
                for xyxy, track_id in zip(bbox_xyxy, identities):
                    if track_id < 0 or track_id is None:
                        continue
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    lines[g_id].append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
        self.videowriter.release()
		for g_id in range(1):
            os.makedirs(os.path.join(self.predict_path+'%d'%g_id), exist_ok=True)
            with open(os.path.join(self.predict_path+'%d'%g_id, f'{self.seq_num}.txt'), 'w') as f:
                f.writelines(lines[g_id])
            print("{}: totally {} dts {} occlusion dts".format(self.seq_num, total_dts[g_id], total_occlusion_dts[g_id]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    # detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    # '''for MOT17 submit''' 
    # sub_dir = 'dancetrack/test' 
    sub_dir = 'dancetrack/val'
    # sub_dir = 'dancetrack/train'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    for ith, vid in enumerate(vids):
        det = Detector(args, model=detr, vid=vid)
        det.detect(args.score_threshold, vis=False)
        # break

    # for g_id in range(args.g_size):
    #     os.system("python TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER ~/yanfeng/data/dancetrack/val  --SEQMAP_FILE ~/yanfeng/data/dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER %s"%(det.predict_path+'%d'%g_id))
    import sys
    sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/MOTRv2/MOTRv3/TrackEval/scripts")
    import run_mot_challenge
    for g_id in range(1):
        res_eval = run_mot_challenge.main(SPLIT_TO_EVAL="val",
                    METRICS=['HOTA', 'CLEAR', 'Identity'],
                    GT_FOLDER="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val",
                    SEQMAP_FILE="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val_seqmap.txt",
                    SKIP_SPLIT_FOL=True,
                    TRACKERS_TO_EVAL=[''],
                    TRACKER_SUB_FOLDER='',
                    USE_PARALLEL=True,
                    NUM_PARALLEL_CORES=8,
                    PLOT_CURVES=False,
                    TRACKERS_FOLDER="%s"%(det.predict_path+'%d'%g_id)
                    )
        print(float(res_eval[0]['MotChallenge2DBox']['']['COMBINED_SEQ']['pedestrian']['summaries'][0]['HOTA']))