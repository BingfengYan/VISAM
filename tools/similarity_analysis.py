import os
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA


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



root_data = 'tmp'

# det2trk_weight = defaultdict(list)
# trk2trk_weight = defaultdict(list)
# detall2trk_weight = defaultdict(list)
# for i in range(703):
#     print(i)
#     for j in range(6):

#         bboxes = np.load(os.path.join(root_data, 'box_%08d_%d.txt.npy'%(i,j)))[0]
#         classes = np.load(os.path.join(root_data, 'class_%08d_%d.txt.npy'%(i,j)))[0, :, 0]
#         weights = np.load(os.path.join(root_data, 'weight_%08d_%d.txt.npy'%(i,j)))

#         bboxes[:, [0,1]] -= bboxes[:, [2,3]]/2
#         bboxes[:, [2,3]] += bboxes[:, [0,1]]

#         indexes = np.where(classes>0)[0]

#         det_indexes = indexes[indexes<60]
#         trk_indexes = indexes[indexes>=60]

#         iou = bboxes_iou(bboxes[trk_indexes], bboxes[det_indexes])
#         if len(trk_indexes) and len(det_indexes):
#             pair_idx = iou.argmax(-1)
#             pair_val = iou.max(-1)
#             pair_trk_idx = trk_indexes[pair_val>0.7]
#             pair_det_idx = det_indexes[pair_idx[pair_val>0.7]]
#             if len(pair_trk_idx) and len(pair_det_idx):
#                 if weights[pair_trk_idx, pair_det_idx].mean() < 1:
#                     det2trk_weight[j].append(weights[pair_trk_idx, pair_det_idx].mean())
#                 else:
#                     print("1")
#                 if weights[pair_trk_idx, pair_trk_idx].mean() < 1:
#                     trk2trk_weight[j].append(weights[pair_trk_idx, pair_trk_idx].mean())
#                 else:
#                     print("1")
#                 if weights[pair_trk_idx, :60].sum(-1).mean() < 1:
#                     detall2trk_weight[j].append(weights[pair_trk_idx, :60].sum(-1).mean())
#                 else:
#                     print("1")

# print(np.array(det2trk_weight[0]).mean(), np.array(det2trk_weight[1]).mean(), np.array(det2trk_weight[2]).mean(), np.array(det2trk_weight[3]).mean(), np.array(det2trk_weight[4]).mean(), np.array(det2trk_weight[5]).mean())
# print(np.array(trk2trk_weight[0]).mean(), np.array(trk2trk_weight[1]).mean(), np.array(trk2trk_weight[2]).mean(), np.array(trk2trk_weight[3]).mean(), np.array(trk2trk_weight[4]).mean(), np.array(trk2trk_weight[5]).mean())
# print(np.array(detall2trk_weight[0]).mean(), np.array(detall2trk_weight[1]).mean(), np.array(detall2trk_weight[2]).mean(), np.array(detall2trk_weight[3]).mean(), np.array(detall2trk_weight[4]).mean(), np.array(detall2trk_weight[5]).mean())

hs_all = defaultdict(list)
hs_all_flatten = []
for i in range(703):
    scores = np.load(os.path.join(root_data, 'class_%08d_%d.txt.npy'%(i,5)))[0, :, 0]
    ids = np.load(os.path.join(root_data, 'ids_%08d.txt.npy'%(i)))[scores>0]
    hs = np.load(os.path.join(root_data, 'hs_%08d.txt.npy'%(i)))[scores>0]
    
    for id, h in zip(ids, hs):
        hs_all[id].append(h)
        hs_all_flatten.append(h)
 
pca = PCA(n_components=2)
# newX = pca.fit_transform(X)
pca.fit(hs_all_flatten) 
pca.transform(X)





stat_scores_det = defaultdict(lambda: defaultdict(int))
for line in np.loadtxt('tmp_det.txt'):
	stat_scores_det[int(line[0])][int(line[1])] = line[2]
stat_scores_trk = defaultdict(lambda: defaultdict(int))
for line in np.loadtxt('tmp_trk.txt'):
	stat_scores_trk[int(line[0])][int(line[1])] = line[2]
stat_scores_uni_det = defaultdict(lambda: defaultdict(int))
for line in np.loadtxt('tmp_uni_det.txt'):
	stat_scores_uni_det[int(line[0])][int(line[1])] = line[2]
stat_scores_uni_trk = defaultdict(lambda: defaultdict(int))
for line in np.loadtxt('tmp_uni_trk.txt'):
	stat_scores_uni_trk[int(line[0])][int(line[1])] = line[2]
 
 
count_bin_all = defaultdict(list)
count_bin = defaultdict(int)
for framid in stat_scores_trk:
	for obj_id in stat_scores_trk[framid]:
		if framid in stat_scores_uni_trk and obj_id in stat_scores_uni_trk[framid]:
			count_bin_all[int(stat_scores_trk[framid][obj_id]*10)].append(stat_scores_uni_trk[framid][obj_id]-stat_scores_trk[framid][obj_id])
			if stat_scores_trk[framid][obj_id] > stat_scores_uni_trk[framid][obj_id]:
				count_bin[int(stat_scores_trk[framid][obj_id]*10)] -= 1
			else:
				count_bin[int(stat_scores_trk[framid][obj_id]*10)] += 1
for i in range(10):
    print(np.array(count_bin_all[i]).mean(), np.array(count_bin_all[i]).std())

   
with open('tmp.txt', 'w') as fp: 
	for framid in stat_scores_trk:
		for obj_id in stat_scores_trk[framid]:
			if framid in stat_scores_uni_trk and obj_id in stat_scores_uni_trk[framid]:
				# print(stat_scores_trk[framid][obj_id], stat_scores_uni_trk[framid][obj_id])
				fp.write('%f %f\n'%(stat_scores_trk[framid][obj_id], stat_scores_uni_trk[framid][obj_id]))
	