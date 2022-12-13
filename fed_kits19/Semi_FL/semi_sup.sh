#!/usr/bin/env bash
###sh semi_sup.sh nnunet 1 2 3e-4 1 '-' 0 10 kits19 50 1 teacher_training 2 3 200 1
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
CLIENT_NUM=${13}
GPU_NUM_PER_SERVER=${14}
ROUND=${15}
IS_PRE_TRAINED=${16}
TEACHER_RUN_ID=${17}
TEACHER_LR=${18}
TEACHER_COMM_ROUND=${19}
TEACHER_EPOCHS=${20}
TH_APPLY=${21}
TH_TYPE=${22}
WEIGT_SCHEME=${23}



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
# sh semi_sup.sh nnunet 0 2 0.0003 1000 '../../../kits19_data/kits19/data/' 0 0 kits19
PROCESS_NUM=`expr $CLIENT_NUM + 1`
echo $PROCESS_NUM
hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./semi_FL.py \
  --model $MODEL \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --gpu $STARTING_GPU\
  --starting_gpu $STARTING_GPU\
  --lr $LEARNING_RATE \
  --comm_round $ROUND \
  --data_dir $DATA_DIR \
  --is_debug $DEBUG \
  --threshold $THRESHOLD \
  --dataset $DATASET \
  --run_id $RUN_ID \
  --local_epoch $LOCAL_EPOCH \
  --phase $PHASE \
  --client_number $CLIENT_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $CLIENT_NUM \
  --is_pre_trained $IS_PRE_TRAINED \
  --teacher_run_id $TEACHER_RUN_ID \
  --teacher_lr $TEACHER_LR\
  --teacher_comm_round $TEACHER_COMM_ROUND\
  --teacher_epoches $TEACHER_EPOCHS\
  --apply_threshold $TH_APPLY\
  --threshold_type $TH_TYPE\
  --weight_scheme $WEIGT_SCHEME


  #sh semi_sup.sh nnunet 1 2 3e-6 1 '-' 0 10 kits19 51 1 student_training 6 7 2000 0 50 3e-4 500 1 True constant org_weight
  #sh semi_sup.sh nnunet 1 2 3e-4 1 '-' 0 10 kits19 52 1 student_training 6 7 2000 0 50 3e-4 500 1 True constant expo_weight
  #sh semi_sup.sh nnunet 1 2 3e-4 1 '-' 0 10 kits19 53 1 student_training 6 7 2000 0 50 3e-4 500 1 False constant dr_weight


