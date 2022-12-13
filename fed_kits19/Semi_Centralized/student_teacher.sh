#!/usr/bin/env bash

MODEL=$1
STARTING_GPU=$2
BATCH_SIZE=$3
LEARNING_RATE=$4
EPOCH=$5
DATA_DIR=$6
DEBUG=$7
THRESHOLD=$8
DATASET=$9
RUN_ID=${10}
LOCAL_EPOCH=${11}
PHASE=${12}
IS_PRETRAINED=${13}
SSL_METHOD=${14}
NNUNET_TR=${15}
# sh student_teacher.sh nnunet 1 2 0.0003 3000 '-' 0 10 kits19 1202 1 'semi_training' 0 'fixmatch' 0


# sh student_teacher.sh nnunet 1 2 0.0003 3000 '-' 0 10 kits19 1201 1 'semi_training' 0 'mixup' 1
# sh student_teacher.sh nnunet 1 2 0.0003 3000 '-' 0 10 kits19 1202 1 'semi_training' 0 'fixmatch' 0

    #data_preprocessing directory
    #if GPU
    # data_dir = '../../kits19_data/kits19/data/'
    #if CPU
    # data_dir = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/data/'

#sh run_kits19.sh 'UNet' 0 4 0.1 100 'sgd' '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/data/' 0
#sh run_kits19.sh 'UNet' 1 1 0.3 100 'sgd' '../../../kits19_data/kits19/data/' 0
#sh kits19_FL.sh 'UNet' 1 1 0.3 100 'sgd' '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/data/' 0 dice both
#sh kits19_FL.sh 'UNet' 7 1 0.3 100 'sgd' '../../../kits19_data/kits19/data/' 0 dice both


######### Centralized #########

#sh semi_sup.sh nnunet 0 2 0.0003 1000 '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/data/' 0 5 kits19
#sh semi_sup.sh nnunet 0 2 0.0003 1000 '../../../kits19_data/kits19/data/' 0 10 kits19

# sh student_teacher.sh nnunet 0 2 0.0003 2000 '-' 0 10 kits19 1 1 'training'
# sh student_teacher.sh nnunet 0 2 0.003 2000 '-' 0 10 kits19 2 1 'training'
# sh student_teacher.sh nnunet 0 2 0.03 2000 '-' 0 10 kits19 3 1 'training'
# sh student_teacher.sh nnunet 0 2 0.3 2000 '-' 0 10 kits19 4 1 'training'

# for more rounds
# sh student_teacher.sh nnunet 6 2 0.0003 3000 '-' 0 10 kits19 9 1 'semi_training' 1
# sh student_teacher.sh nnunet 7 2 0.0003 3000 '-' 0 10 kits19 10 1 'sup_training' 1

#Loss Change
# sh student_teacher.sh nnunet 3 2 0.0003 3000 '-' 0 10 kits19 10 1 'semi_training' 1

# mpirun -np 1 -hostfile ./mpi_host_file python3 ./student_teacher_semi.py \
hostname > mpi_host_file

mpirun -np 1 -hostfile ./mpi_host_file python3 ./cutmix_centralized.py \
  --model $MODEL \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --gpu $STARTING_GPU\
  --starting_gpu $STARTING_GPU\
  --lr $LEARNING_RATE \
  --data_dir $DATA_DIR \
  --is_debug $DEBUG \
  --threshold $THRESHOLD \
  --dataset $DATASET \
  --run_id $RUN_ID \
  --local_epoch $LOCAL_EPOCH \
  --phase $PHASE \
  --is_pre_trained $IS_PRETRAINED \
  --ssl_method $SSL_METHOD \
  --nnunet_tr $NNUNET_TR

