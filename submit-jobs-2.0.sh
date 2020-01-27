#!/bin/sh

# Hard coded settings for resources
# time limit
export ttime=4:00
# number of gpus per job
export num_gpu_per_job=1
# memory per job
export mem_per_gpu=30000

export JOB_NAME='elliot_test'

# load python
module load eth_proxy python_gpu/3.6.4
module load cuda/10.0.130
module load cudnn/7.6.4


export CONFIG="/cluster/home/ebeck/ResNet50/configs/config_class_resnet.json"
export L2_PEN=0.001
export DO_RATE=0.5
export DECAY_EVERY=5000
export NUM_EPOCHS=10
export BATCH_SIZE=32


for VAR_LEARN_RATE in .01 .001 .0001 .00001
    do
        export LEARN_RATE=$VAR_LEARN_RATE
        sh submit-train-2.0.sh
done