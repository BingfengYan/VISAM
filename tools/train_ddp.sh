#!/usr/bin/env bash
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


cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import util.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
echo "worker list command is $worker_list_command"
eval worker_list=`python -c "$worker_list_command"`
echo "worker list is $worker_list"
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
index_command="import util.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
eval node_rank=`python -c "$index_command"`
echo "node rank is $node_rank"
dist_url="tcp://$master_addr:$master_port"
echo "dist url is $dist_url"
PYTHONPATH=$PYTHONPATH:../ \
# python tools/run_net.py \
#    --num_shards 8 \
#    --shard_id $node_rank \
#    --dist_url $dist_url \
#    --cfg configs/verb/MVIT_B_32x2_CONV.yaml

MASTER_ADDR=${MASTER_ADDR:-$master_addr}
MASTER_PORT=${MASTER_PORT:-$master_port}
NODE_RANK=${NODE_RANK:-$node_rank}
# let "NNODES=GPUS/GPUS_PER_NODE"

NODE_NUM=${#worker_strs[@]}  
echo "node num is $NODE_NUM"

if ((NODE_RANK == 0)); then
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

  pushd $OUTPUT_DIR
  trap cleanup EXIT

  # log git status
  echo "Logging git status"
  git status > git_status
  git rev-parse HEAD > git_tag
  git diff > git_diff
  echo $PY_ARGS > desc
  echo " ...Done"

else
  # 3 minutes
  sleep 180
  for RUN in $(seq 100); do
    ls $OUTPUT_BASE | grep run$RUN && continue
    let "ITERRUN=$RUN-1"
    OUTPUT_DIR=$OUTPUT_BASE/run$ITERRUN
    break
  done
fi

args=$(cat $1)

# python -m torch.distributed.launch --nproc_per_node=8   --master_port 29502 --use_env main.py ${args} --output_dir $OUTPUT_DIR

# python ./util/launch.py \
#     --nnodes 2 \
#     --node_rank ${NODE_RANK} \
#     --master_addr ${MASTER_ADDR} \
#     --master_port 29502 \
#     --nproc_per_node 8 \
#     python main.py "${args} --output_dir $OUTPUT_DIR"
python -m torch.distributed.launch --nproc_per_node=8 --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port 29502 --use_env main.py ${args} --output_dir $OUTPUT_DIR
