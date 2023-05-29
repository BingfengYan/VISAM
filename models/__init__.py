# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_transformer_plus import DeformableTransformer
from .deformable_transformer_cross import DeformableTransformer as DeformableTransformerCross
from .ftransformer import DetrTransformerDecoder
def build_deforamble_transformer(args):
    arch_catalog = {
        'DeformableTransformer': DeformableTransformer,
        'DeformableTransformerCross': DeformableTransformerCross,
    }
    assert args.trans_mode in arch_catalog, 'invalid arch: {}'.format(args.trans_mode)
    build_func = arch_catalog[args.trans_mode]
    
    return build_func(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        memory_bank=args.memory_bank_type == 'MemoryBankFeat'
    )
    

from .motr import build as build_motr
from .motr_co import build as build_motr_uninCost



def build_model(args):
    arch_catalog = {
        'motr': build_motr,
        
        'motr_unincost': build_motr_uninCost,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)


