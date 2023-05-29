'''
Author: 颜峰 && bphengyan@163.com
Date: 2023-05-18 12:51:56
LastEditors: 颜峰 && bphengyan@163.com
LastEditTime: 2023-05-18 12:51:57
FilePath: /CO-MOT/datasets/__init__.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .dance import build as build_e2e_dance


def build_dataset(image_set, args):
    if args.dataset_file == 'e2e_dance':
        return build_e2e_dance(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
