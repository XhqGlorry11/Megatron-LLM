#!/bin/bash

chmod +x ./s5cmd

pushd megatron/data/
  make
popd

DATA_PATH=/opt/ml/input/data/fsx
echo "show fsx dataset: ls -alh ${DATA_PATH}"
ls -alh $DATA_PATH

CHECKPOINT_PATH=/tmp/pretrained
checkpoint_iter=iter_0118000
s3_checkpoint_txt_path="${S3_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt"
s3_checkpoint_model_path="${S3_CHECKPOINT_PATH}/${checkpoint_iter}/*"
mkdir ${CHECKPOINT_PATH}/${checkpoint_iter}
./s5cmd sync $s3_checkpoint_txt_path $CHECKPOINT_PATH
./s5cmd sync $s3_checkpoint_model_path ${CHECKPOINT_PATH}/${checkpoint_iter}
echo "show checkpoint data: ls -alh ${CHECKPOINT_PATH}"
ls -alh $CHECKPOINT_PATH
echo "show checkpoint data: ls -alh ${CHECKPOINT_PATH}/${checkpoint_iter}"
ls -alh ${CHECKPOINT_PATH}/${checkpoint_iter}




mkdir /opt/conda/lib64
cp /opt/conda/lib/libcudart.so /opt/conda/lib64/libcudart.so

df -h


WORKING_DIR=/opt/ml/code
# SM_WORKING_DIR=/tmp/output
SM_WORKING_DIR=/opt/ml/checkpoints # aws checkpoint dir auto save to s3, see start_sage_env.py: checkpoint_s3_uri


#The related information about multi-nodes cluster.
MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"
GPUS_PER_NODE="$SM_NUM_GPUS"
SAVE_PATH="${SM_WORKING_DIR}/${JOB_NAME}"
# LOAD_PATH="${SM_WORKING_DIR}/${JOB_NAME}"
LOAD_PATH=$CHECKPOINT_PATH

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

PARALELLE_ARGS="--tensor_model_parallel_size 1 \
				--pipeline_model_parallel_size 1"

DATA_ARGS="--data_path ${DATA_PATH}/4_200B \
			--tokenizer_type TsTokenizer \
			--vocab_file ${WORKING_DIR}/tokenizer/TsTokenizer_dustbin \
			--num_workers 64 \
			--no_new_tokens \
			--make_vocab_size_divisible_by 128 \
			--seq_length 4096 \
			--max_position_embeddings 4096"

MODEL_ARGS="--model_name llama2 \
			--use_gpt_neox_init_method \
			--use_gpt_neox_output_layer_init_method
			--use_flash_attn \
			--sequence_parallel \
			--recompute_granularity selective \
			--num_layers 24 \
			--hidden_size 2048 \
			--num_attention_heads 16 \
			--ffn_hidden_size 5632 \
			--position_embedding_type rotary \
			--glu_activation swiglu \
			--use_rms_norm \
			--layernorm_epsilon 0.00001 \
			--no_tie_embed_logits \
			--no_bias_gelu_fusion \
			--no_bias_dropout_fusion \
			--hidden_dropout 0.0 \
			--attention_dropout 0.0"

TRAIN_ARGS="--fp16 \
			--initial_loss_scale 65536 \
			--loss_scale_window 500 \
			--clip_grad 1.0 \
			--micro_batch_size 3 \
			--global_batch_size 768 \
			--use_checkpoint_args \
			--use_checkpoint_opt_param_scheduler \
			--train_iters 600000 \
			--lr_decay_style cosine \
			--lr_warmup_iters 1000 \
			--lr 0.0003 \
			--min_lr 0.00003 \
			--weight_decay 0.1"

LOG_SAVE_ARGS="--save ${SAVE_PATH} \
			--load ${LOAD_PATH} \
			--log_interval 10 \
			--save_interval 5000 \
			--eval_interval 1000 \
			--eval_iters 100 \
			--log_params_norm \
			--log_timers_to_tensorboard \
			--wandb_logger \
			--wandb_project llama2-1.3B-AWS \
			--wandb_id 3rd_200B \
			--wandb_api_key fac46169cec8e164a47ed1c71199e3e8e9f02cc5"

CMD="torchrun ${DISTRIBUTED_ARGS} ${WORKING_DIR}/finetune.py ${PARALELLE_ARGS} ${DATA_ARGS} ${MODEL_ARGS} ${TRAIN_ARGS} ${LOG_SAVE_ARGS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/aws_training_log.txt
echo "training finished, all checkpoint saved to"$SAVE_PATH

###########################################################
# delete megatron/data/helper.cyt* in advance. !!!!!
###########################################################
