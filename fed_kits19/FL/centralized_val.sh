#!/usr/bin/env bash

#KITS19
#sh fedavg_kits19.sh 2 2 1 2 nnunet equal 100 1 2 0.003 kits19 "./../../../data/kits19" adam 0

#perFedAvg
#sh run_fedavg_distributed_pytorch.sh 8 8 1 4 resnet18 lda 1000 1 32 0.003 cifar10 "./../../../data/cifar10" adam 0 perFedAvg 5000 0 0.1
#sh run_fedavg_distributed_pytorch.sh 8 8 1 4 resnet18 lda 1000 1 32 0.001 cifar10 "./../../../data/cifar10" adam 0 perFedAvg 5001 4 0.1

#sh centralized_val.sh 1 1 1 nnunet 3000 1 3e-4 kits19 0 5 training 1 0
CLIENT_NUM=$1
SERVER_NUM=$2
GPU_NUM_PER_SERVER=$3
MODEL=$4
ROUND=$5
EPOCH=$6
LR=$7
DATASET=$8
GPU_STARTING_FROM=$9
DEBUG=${10}
PHASE=${11}
RUN_ID=${12}
IS_PRETRAINED=${13}



PROCESS_NUM=`expr $CLIENT_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np 1 -hostfile ./mpi_host_file python3 ./validation_centralized.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --client_num_in_total $CLIENT_NUM \
  --client_number  $CLIENT_NUM\
  --client_num_per_round $CLIENT_NUM \
  --epochs $ROUND \
  --local_epoch $EPOCH \
  --starting_gpu $GPU_STARTING_FROM \
  --lr $LR \
  --is_debug $DEBUG \
  --phase $PHASE \
  --run_id $RUN_ID \
  --is_pre_trained $IS_PRETRAINED

