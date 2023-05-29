#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

# args=$(cat configs/motrv2.args)
args=$(cat $1)
python3 submit_mot.py ${args} --exp_name tracker --resume $2 $3
