# CO-MOT: Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking


[![arXiv]](https://arxiv.org/abs/2305.12724)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-end-to-end-and-non/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=bridging-the-gap-between-end-to-end-and-non)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-end-to-end-and-non/multi-object-tracking-on-bdd100k)](https://paperswithcode.com/sota/multi-object-tracking-on-bdd100k?p=bridging-the-gap-between-end-to-end-and-non)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-end-to-end-and-non/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bridging-the-gap-between-end-to-end-and-non)
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motrv2-bootstrapping-end-to-end-multi-object/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=motrv2-bootstrapping-end-to-end-multi-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motrv2-bootstrapping-end-to-end-multi-object/multiple-object-tracking-on-bdd100k)](https://paperswithcode.com/sota/multiple-object-tracking-on-bdd100k?p=motrv2-bootstrapping-end-to-end-multi-object) -->

This repository is an official implementation of [CO-MOT](https://arxiv.org/abs/2305.12724).

**TO DO**
1. release code of grounded MOT.

## Introduction

Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking.

<!-- ![Overview](https://raw.githubusercontent.com/zyayoung/oss/main/motrv2_main.jpg) -->

**Abstract.** Existing end-to-end Multi-Object Tracking (e2e-MOT) methods have not surpassed non-end-to-end tracking-by-detection methods. One potential reason is its label assignment strategy during training that consistently binds the tracked objects with tracking queries and then assigns the few newborns to detection queries. With one-to-one bipartite matching, such an assignment will yield unbalanced training, i.e., scarce positive samples for detection queries, especially for an enclosed scene, as the majority of the newborns come on stage at the beginning of videos. Thus, e2e-MOT will be easier to yield a tracking terminal without renewal or re-initialization, compared to other tracking-by-detection methods. To alleviate this problem, we present Co-MOT, a simple and effective method to facilitate e2e-MOT by a novel coopetition label assignment with a shadow concept. Specifically, we add tracked objects to the matching targets for detection queries when performing the label assignment for training the intermediate decoders. For query initialization, we expand each query by a set of shadow counterparts with limited disturbance to itself. With extensive ablations, Co-MOT achieves superior performance without extra costs, e.g., 69.4% HOTA on DanceTrack and 52.8% TETA on BDD100K. Impressively, Co-MOT only requires 38\% FLOPs of MOTRv2 to attain a similar performance, resulting in the 1.4× faster inference speed.

<!-- ## News
- **2022.11.18** MOTRv2 paper is available on [arxiv](https://arxiv.org/abs/2211.09791).
- **2022.10.27** Our DanceTrack challenge tech report is released [[arxiv]](https://arxiv.org/abs/2210.15281) [[ECCVW Challenge]](https://motcomplex.github.io/index.html#challenge).
- **2022.10.05** MOTRv2 achieved the 1st place in the [1st Multiple People Tracking in Group Dance Challenge](https://motcomplex.github.io/). -->

## Main Results

### DanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   69.9   |   82.1   |   58.9   |   91.2   |   71.9   | [model](https://drive.google.com/file/d/1rwUpcyufIMdfSIes5esytMk_Phn3i-3b/view?usp=share_link) |


|VISAM|
|![](https://raw.githubusercontent.com/BingfengYan/MOTSAM/main/visam.gif)|


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MOTR](https://github.com/megvii-research/MOTR).

### Requirements

* Install pytorch using conda (optional)

    ```bash
    conda create -n motrv2 python=3.7
    conda activate motrv2
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

1. Please download [DanceTrack](https://dancetrack.github.io/) and [CrowdHuman](https://www.crowdhuman.org/) and unzip them as follows:

```
/data/Dataset/mot
├── crowdhuman
│   ├── annotation_train.odgt
│   ├── annotation_trainval.odgt
│   ├── annotation_val.odgt
│   └── Images
├── DanceTrack
│   ├── test
│   ├── train
│   └── val
├── det_db_motrv2.json
```

You may use the following command for generating crowdhuman trainval annotation:

```bash
cat annotation_train.odgt annotation_val.odgt > annotation_trainval.odgt
```

### Training

You may download the coco pretrained weight from [Deformable DETR (+ iterative bounding box refinement)](https://github.com/fundamentalvision/Deformable-DETR#:~:text=config%0Alog-,model,-%2B%2B%20two%2Dstage%20Deformable), and modify the `--pretrained` argument to the path of the weight. Then training MOTR on 8 GPUs as following:

```bash 
./tools/train.sh configs/motrv2ch_uni5cost3ggoon.args
```

### Inference on DanceTrack Test Set

```bash
# run a simple inference on our pretrained weights
./tools/simple_inference.sh configs/motrv2ch_uni5cost3ggoon.args ./motrv2_dancetrack.pth

# Or evaluate an experiment run
# ./tools/eval.sh exps/motrv2/run1

# then zip the results
zip motrv2.zip tracker/ -r
```

## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [BDD100K](https://github.com/bdd100k/bdd100k)
- [MOTRv2](https://github.com/megvii-research/MOTRv2)
- [CO-MOT](https://github.com/BingfengYan/CO-MOT)
