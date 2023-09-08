#!/bin/bash

echo 'printenv | grep NCCL'
printenv | grep NCCL


echo 'echo $CUDA_HOME'
echo $CUDA_HOME
echo 'ls CUDA_HOME'
ls $CUDA_HOME


echo 'echo $LD_LIBRARY_PATH'
echo $LD_LIBRARY_PATH
echo 'ls $LD_LIBRARY_PATH'
ls $LD_LIBRARY_PATH

echo 'ip -a'
ip -a


#echo 'find  libcudart.so'
#find / -name  libcudart.so

chmod +x ./s5cmd
#python -m pip install packaging
#python -m pip install -r requirements.txt
#python -m pip install wandb
git clone https://github.com/NVIDIA/apex
pushd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
#git clone https://github.com/NVIDIA/apex
#pushd apex
#pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
#popd
pushd megatron/data/
  make
popd
#python -m pip install .


# echo "Downloading: $S3_DATA_PATH"
# ./s5cmd sync $S3_DATA_PATH /tmp/train_data
# echo "ls -al /tmp/train_data"
# ls -al /tmp/train_data

DATA_PATH=/opt/ml/input/data/fsx
echo "show fsx datas: ls -alh ${DATA_PATH}"
ls -alh $DATA_PATH



LOAD_PATH=/tmp/pretrained
./s5cmd sync $S3_MODEL_PATH $LOAD_PATH
echo "ls -al ${LOAD_PATH}"
ls -al $LOAD_PATH

mkdir /opt/conda/lib64
cp /opt/conda/lib/libcudart.so /opt/conda/lib64/libcudart.so

df -h


WORKING_DIR=/opt/ml/code
# SM_WORKING_DIR=/tmp/output
SM_WORKING_DIR=/opt/ml/checkpoints # aws checkpoint dir auto save to s3, see start_sage_env.py: checkpoint_s3_uri

echo "copy tokenizer to ${SM_WORKING_DIR}"
cp $DATA_PATH/tokenizer $SM_WORKING_DIR
ls -al $SM_WORKING_DIR

#The related information about multi-nodes cluster.
MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"
GPUS_PER_NODE="$SM_NUM_GPUS"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
JOB_NAME="$JOB_NAME"
SAVE_PATH="${SM_WORKING_DIR}/results"
LOG_FILE="${SAVE_PATH}/log.txt"

#GLOBAL_BATCH_SIZE=32
#SAVE_INTERVAL=200
#TRAIN_ITERS=600
#LR_WARMUP_ITERS=50
#LOG_INTERVAL=2
#
#GLOBAL_BATCH_SIZE=64
#SAVE_INTERVAL=100
#TRAIN_ITERS=500
#LR_WARMUP_ITERS=50
#LOG_INTERVAL=10

GLOBAL_BATCH_SIZE=1024
SAVE_INTERVAL=2000
TRAIN_ITERS=62000
LR_WARMUP_ITERS=2000
LOG_INTERVAL=10


SCRIPT_CONFIG="--tensor_model_parallel_size 4 \
	--pipeline_model_parallel_size 2 \
  --load ${LOAD_PATH} \
	--save ${SAVE_PATH} \
	--tensorboard_dir ${SAVE_PATH}/tensorboard/ \
	--data_path ${DATA_PATH}/train_data_text_document \
	--model_name llama2 \
	--tokenizer_type TsTokenizer \
	--vocab_file=${DATA_PATH}/tokenizer \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 2 \
	--global_batch_size ${GLOBAL_BATCH_SIZE} \
	--sequence_parallel \
	--recompute_granularity selective \
  --wandb_logger \
	--wandb_project llama2-continue-pretrain \
  --wandb_entity linyizuo \
	--use_checkpoint_args"


LOG_ARGS="--log_interval ${LOG_INTERVAL} --save_interval ${SAVE_INTERVAL} --eval_interval 62000"
TRAIN_ARGS="--train_iters ${TRAIN_ITERS} --eval_iters 0 --lr_decay_style cosine --lr_warmup_iters ${LR_WARMUP_ITERS} --lr 4e-5 --min_lr 4e-6"
LLAMA_ARGS="--num_layers 32 --use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

CMD="torchrun ${DISTRIBUTED_ARGS} ${WORKING_DIR}/finetune.py ${SCRIPT_CONFIG} ${COMMON_ARGS} ${LOG_ARGS} ${TRAIN_ARGS} ${LLAMA_ARGS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
echo "traing finished, all checkpoint saved to"$SAVE_PATH
# echo "uploading ckpt"
# ./s5cmd sync ${SM_WORKING_DIR} ${S3_SAVE_PATH}
# echo "End over"