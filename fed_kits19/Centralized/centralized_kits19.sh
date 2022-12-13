#!/usr/bin/env bash

MODEL=$1
STARTING_GPU=$2
LEARNING_RATE=$3
EPOCH=$4
DEBUG=$5
DATASET=$6
RUN_ID=$7
LOCAL_EPOCH=$8
PHASE=$9
PRE_TRAIN=${10}


hostname > mpi_host_file

mpirun -np 1 -hostfile ./mpi_host_file python3 ./centralized.py \
  --model $MODEL \
  --epochs $EPOCH \
  --gpu $STARTING_GPU\
  --starting_gpu $STARTING_GPU\
  --lr $LEARNING_RATE \
  --is_debug $DEBUG \
  --dataset $DATASET \
  --run_id $RUN_ID \
  --local_epoch $LOCAL_EPOCH \
  --phase $PHASE \
  --is_pre_trained $PRE_TRAIN

# sh centralized_kits19.sh nnunet 0 0.0003 3000 0 kits19 600 1 train
# sh centralized_kits19.sh nnunet 0 0.0003 3000 0 kits19 1105 1 train 0