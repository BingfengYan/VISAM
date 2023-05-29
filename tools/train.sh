#!/usr/bin/env bash
###
 # @Author: 颜峰 && bphengyan@163.com
 # @Date: 2023-05-19 17:19:11
 # @LastEditors: 颜峰 && bphengyan@163.com
 # @LastEditTime: 2023-05-22 09:54:42
 # @FilePath: /CO-MOT/tools/train.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
### 
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

# 打印所有指令
set -x

PY_ARGS=${@:2}

# 脚本运行失败，报错
set -o pipefail
#sed -e  ：直接在指令列模式上進行 sed 的動作編輯；
OUTPUT_BASE=$(echo $1 | sed -e "s/configs/exps/g" | sed -e "s/.args$//g")
mkdir -p $OUTPUT_BASE

for RUN in $(seq 100); do
  ls $OUTPUT_BASE | grep run$RUN && continue
  OUTPUT_DIR=$OUTPUT_BASE/run$RUN
  mkdir $OUTPUT_DIR && break
done

# clean up *.pyc files
rmpyc() {
  rm -rf $(find -name __pycache__)
  rm -rf $(find -name "*.pyc")
}

# run backup
echo "Backing up to log dir: $OUTPUT_DIR"
rmpyc && cp -r models datasets util main.py engine.py eval_detr.py seqmap submit_dance.py $1 $OUTPUT_DIR
echo " ...Done"

# tar src to avoid future editing
cleanup() {
  echo "Packing source code"
  rmpyc
  # tar -zcf models datasets util main.py engine.py eval.py submit.py --remove-files
  echo " ...Done"
}

args=$(cat $1)

pushd $OUTPUT_DIR
trap cleanup EXIT

# log git status
echo "Logging git status"
git status > git_status
git rev-parse HEAD > git_tag
git diff > git_diff
echo $PY_ARGS > desc
echo " ...Done"

python -m torch.distributed.launch --nproc_per_node=4   --master_port 29504 --use_env main.py ${args} --output_dir $OUTPUT_DIR |& tee -a output.log
