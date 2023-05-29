# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch.distributed as dist
import torch
import util.misc as utils

from datasets.data_prefetcher import data_dict_to_cuda
attr_dict = dict()
attr_dict["categories"] = {
    0: {"supercategory": "none", "id": 0, "name": "pedestrian"},
    1: {"supercategory": "none", "id": 1, "name": "bicycle"},
    2: {"supercategory": "none", "id": 2, "name": "car"},
    3: {"supercategory": "none", "id": 3, "name": "motorcycle"},
    5: {"supercategory": "none", "id": 5, "name": "bus"},
    6: {"supercategory": "none", "id": 6, "name": "train"},
    7: {"supercategory": "none", "id": 7, "name": "truck"},
    90: {"supercategory": "none", "id": 90, "name": "rider"},
    91: {"supercategory": "none", "id": 91, "name": "other person"},
    92: {"supercategory": "none", "id": 92, "name": "trailer"},
    93: {"supercategory": "none", "id": 93, "name": "other vehicle"}
}

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    iter_num = 0
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if True: 
            loss_two = 0
            loss_ori = 0
            for k, v in loss_dict_reduced_scaled.items():
                if '_two_' in k: loss_two += v
                else: loss_ori += v
            loss_dict_reduced_scaled['loss_ori'] = loss_ori
            loss_dict_reduced_scaled['loss_two'] = loss_two
            # if loss_two > 0:
            #     losses /= 2.0  # 由于多加了一倍的loss，因此这里减掉
            #     loss_value /= 2.0
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        if torch.isnan(grad_total_norm).any():
            print(data_dict['gt_instances'])
            optimizer.zero_grad()
            
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'same_num_dict'):
                    # if True:
            same_num_dict = utils.reduce_dict(criterion.same_num_dict, average=False)
            same = 0
            all = 0
            for k, v in same_num_dict.items():
                if '_same' in k: same += v
                else: all += v
            if all > 0:
                same_num_dict['ratio'] = same * 1.0 / all
            metric_logger.update(loss=loss_value, **dict(loss_dict_reduced_scaled.items(), **same_num_dict))
        else:
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # gather the stats from all processes
        # break
        
        # import numpy as np
        # with open('tmp1/grad_%d.txt'%iter_num,'w') as f:
        #     for name, parms in model.named_parameters():	
        #         if parms.grad is  None: continue
        #         np.savetxt(f, parms.grad.view(-1).cpu().detach().numpy()[:100], delimiter=" ", header=name, comments='//', fmt='%.50f')
        # print(iter_num)
        # iter_num+=1
        # if iter_num > 20:
        #     iter_num = 20
  
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import cv2
import json
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torchvision.transforms.functional as F
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


def filter_dt_by_score(dt_instances, prob_threshold):
    keep = dt_instances.scores > prob_threshold
    keep &= dt_instances.obj_idxes >= 0
    return dt_instances[keep]

def filter_dt_by_area(dt_instances, area_threshold):
    wh = dt_instances.boxes[..., 2:4] - dt_instances.boxes[..., 0:2]
    areas = wh[..., 0] * wh[..., 1]
    keep = areas > area_threshold
    return dt_instances[keep]

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # header = 'Test:'
    # print_freq = 10
    predict_path = os.path.join(output_dir, 'tracker')
    prob_threshold=0.5
    area_threshold=100

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in data_loader:
        print(data_dict)
        # data_dict = data_dict_to_cuda(data_dict, device)
        # outputs = model.inference_single_image (data_dict)
        
        seq_num = os.path.basename(data_dict['video_name'][0])
        if args.dataset_file == 'e2e_bdd' or args.dataset_file == 'e2e_bddcc':
            img_list = os.listdir(data_dict['video_name'][0])
            img_list = [os.path.join(data_dict['video_name'][0], i) for i in img_list if 'jpg' in i]
        else:
            img_list = os.listdir(os.path.join(data_dict['video_name'][0], 'img1'))
            img_list = [os.path.join(data_dict['video_name'][0], 'img1', i) for i in img_list if 'jpg' in i]
        
        img_list = sorted(img_list)
        
        track_instances = None
        det_db = []
        loader = DataLoader(ListImgDataset('', img_list, det_db), 1, num_workers=2)
        lines = defaultdict(list)
        total_dts = defaultdict(int)
        total_occlusion_dts = defaultdict(int)
        # print('g_size: %d'%self.args.g_size)
        for i, data in enumerate(loader):   # tqdm(loader)):
            cur_img, ori_img, proposals, f_path = [d[0] for d in data]
            cur_img, proposals = cur_img.to(device), proposals.to(device)

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                # track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            # 内部包含backboe+encode+decode+跟踪匹配关系+跟踪目标过滤（从query中过滤）
            try: 
                res = model.module.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            except:
                res = model.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances_all = deepcopy(track_instances).get_bn(0)

            # filter det instances by score.
            dt_instances_all = filter_dt_by_score(dt_instances_all, prob_threshold)  # 保留置信度比较高的目标（因为motr内部可能会保留相对置信度高一些的目标，但输出需要输出比较高一些）
            dt_instances_all = filter_dt_by_area(dt_instances_all, area_threshold) # 过滤小目标
            
            active_indx = []
            full_indx = torch.arange(len(dt_instances_all), device=dt_instances_all.scores.device)
            for id in torch.unique(dt_instances_all.obj_idxes):
                indx = torch.where(dt_instances_all.obj_idxes == id)[0]
                active_indx.append(full_indx[indx][dt_instances_all.scores[indx].argmax()])
            if len(active_indx):
                active_indx = torch.stack(active_indx)
                dt_instances_all = dt_instances_all[active_indx]
            
            for g_id in range(args.g_size):
                # dt_instances = dt_instances_all[dt_instances_all.group_ids==g_id]
                dt_instances = dt_instances_all
                
                total_dts[g_id] += len(dt_instances)

                bbox_xyxy = dt_instances.boxes.tolist()
                identities = dt_instances.obj_idxes.tolist()
                labels = dt_instances.labels.tolist()
                if args.dataset_file == 'e2e_bdd' or args.dataset_file == 'e2e_bddcc':
                    labels_tmp = []
                    for xyxy, track_id, category in zip(bbox_xyxy, identities, labels):
                        if track_id < 0 or track_id is None:
                            continue
                        if category not in list(attr_dict["categories"].keys()): continue
                        x1, y1, x2, y2 = xyxy
                        w, h = x2 - x1, y2 - y1
                        labels_tmp.append({"id": str(track_id), 
                                        "category": attr_dict["categories"][category]['name'], 
                                        "attributes": {"crowd": False, "occluded": False, "truncated": False},
                                        "box2d": {"x1": x1, "y1": y1, "x2": x1+w, "y2": y1+h}})
                    save_format = {"name": os.path.basename(f_path), "videoName": os.path.basename(os.path.dirname(f_path)), "frameIndex": i, "labels": labels_tmp}
                    lines[g_id].append(save_format)
                else:
                    save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
                    for xyxy, track_id in zip(bbox_xyxy, identities):
                        if track_id < 0 or track_id is None:
                            continue
                        x1, y1, x2, y2 = xyxy
                        w, h = x2 - x1, y2 - y1
                        if args.dataset_file == 'e2e_mot':
                            frame_ith = int(os.path.splitext(os.path.basename(f_path))[0])
                            lines[g_id].append(save_format.format(frame=frame_ith, id=track_id, x1=x1, y1=y1, w=w, h=h))
                        else:
                            lines[g_id].append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
                    
        for g_id in range(args.g_size):
            os.makedirs(os.path.join(predict_path+'%d'%g_id), exist_ok=True)
            if args.dataset_file == 'e2e_bdd' or args.dataset_file == 'e2e_bddcc':
                with open(os.path.join(predict_path+'%d'%g_id, f'{seq_num}.json'), 'w') as f:
                    json.dump(lines[g_id], f)
            else:
                with open(os.path.join(predict_path+'%d'%g_id, f'{seq_num}.txt'), 'w') as f:
                    f.writelines(lines[g_id])
            print("{}: totally {} dts {} occlusion dts".format(seq_num, total_dts[g_id], total_occlusion_dts[g_id]))

    if dist.is_initialized():
        dist.barrier()
    # if utils.get_local_rank() == 0:
        # for g_id in range(g_size):
        #     os.system("python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/MOTRv2/MOTRv3/TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER /mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val  --SEQMAP_FILE /mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER %s"%(predict_path+'%d'%g_id))
    if args.dataset_file == 'e2e_mot':
        import sys
        sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/MOTRv2/MOTRv3/TrackEval/scripts")
        import run_mot_challenge
        for g_id in range(args.g_size):
            res_eval = run_mot_challenge.main(SPLIT_TO_EVAL="val",
                        METRICS=['HOTA', 'CLEAR', 'Identity'],
                        GT_FOLDER="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/MOT/MOT17/val/",
                        SEQMAP_FILE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/MOT/MOT17/val_seqmap.txt",
                        SKIP_SPLIT_FOL=True,
                        TRACKERS_TO_EVAL=[''],
                        TRACKER_SUB_FOLDER='',
                        USE_PARALLEL=True,
                        NUM_PARALLEL_CORES=8,
                        PLOT_CURVES=False,
                        TRACKERS_FOLDER="%s"%(predict_path+'%d'%g_id)
                        )
        return float(res_eval[0]['MotChallenge2DBox']['']['COMBINED_SEQ']['pedestrian']['summaries'][0]['HOTA'])
    if args.dataset_file == 'e2e_bdd' or args.dataset_file == 'e2e_bddcc':
        import sys
        sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/MOTRv2/MOTRv3/TrackEval/scripts")
        import run_bdd
        for g_id in range(args.g_size):
            # os.system("python TrackEval/scripts/run_bdd.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER /mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/BDD100K/labels/box_track_20/val/   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER %s"%(det.predict_path+'%d'%g_id))

            res_eval = run_bdd.main(SPLIT_TO_EVAL="val",
                        METRICS=['HOTA', 'CLEAR', 'Identity'],
                        GT_FOLDER="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/BDD100K/labels/box_track_20/val/",
                        TRACKERS_TO_EVAL=[''],
                        TRACKER_SUB_FOLDER='',
                        USE_PARALLEL=True,
                        NUM_PARALLEL_CORES=8,
                        PLOT_CURVES=False,
                        TRACKERS_FOLDER="%s"%(predict_path+'%d'%g_id)
                        )
        return float(res_eval[0]['BDD100K']['']['COMBINED_SEQ']['cls_comb_cls_av']['summaries'][0]['HOTA'])
    else:
        import sys
        sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/MOTRv2/MOTRv3/TrackEval/scripts")
        import run_mot_challenge
        for g_id in range(args.g_size):
            res_eval = run_mot_challenge.main(SPLIT_TO_EVAL="val",
                        METRICS=['HOTA', 'CLEAR', 'Identity'],
                        GT_FOLDER="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val",
                        SEQMAP_FILE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/val_seqmap.txt",
                        SKIP_SPLIT_FOL=True,
                        TRACKERS_TO_EVAL=[''],
                        TRACKER_SUB_FOLDER='',
                        USE_PARALLEL=True,
                        NUM_PARALLEL_CORES=8,
                        PLOT_CURVES=False,
                        TRACKERS_FOLDER="%s"%(predict_path+'%d'%g_id)
                        )
        return float(res_eval[0]['MotChallenge2DBox']['']['COMBINED_SEQ']['pedestrian']['summaries'][0]['HOTA'])

