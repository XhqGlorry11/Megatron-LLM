#！ /usr/bin/bash

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export HTTP_PROXY=http://172.17.0.10:10811
export HTTPS_PROXY=http://172.17.0.10:10811
export NO_PROXY=localhost,127.0.0.1,localaddress,.localdomain.com,30.139.132.202/24,172.17.0.1/16
wandb login fac46169cec8e164a47ed1c71199e3e8e9f02cc5


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python -m torch.distributed.launch --nproc_per_node=8 --master_port=58417 --use_env \
torchrun --nproc_per_node 8 --nnodes 3 --node_rank 0 --master_addr 172.17.0.6 --master_port 8088 \
/home/xinghq/megatron-llm/finetune.py \
--tensor_model_parallel_size=1 \
--pipeline_model_parallel_size=1 \
--save=/hstore/llm_train_val/megatron-llm/fintune_llama_fp16_grad1_mtl-initialization/checkpoints \
--load=/hstore/llm_train_val/megatron-llm/fintune_llama_fp16_grad1_mtl-initialization/checkpoints \
--train_data_path=/hstore/llm_data/The_Pile/train/tokenization/train_data_text_document \
--test_data_path=/hstore/llm_data/The_Pile/test/tokenization/test_data_text_document \
--valid_data_path=/hstore/llm_data/The_Pile/validate/tokenization/validate_data_text_document \
--model_name=llama2 \
--tokenizer_type=HFTokenizer \
--vocab_file=/home/xinghq/megatron-llm/tokenizer/neox_20B_tokenizer.json \
--fp16 \
--clip_grad=1 \
--use_flash_attn \
--micro_batch_size=3 \
--global_batch_size=576 \
--sequence_parallel \
--recompute_granularity=selective \
--use_checkpoint_args \
--num_workers=32 \
--train_iters=100000 \
--lr_decay_style=cosine \
--lr_warmup_fraction=0.01 \
--lr=0.0003 \
--min_lr=0.00003 \
--no_new_tokens \
--num_layers=24 \
--hidden_size=2048 \
--num_attention_heads=16 \
--ffn_hidden_size=5632 \
--make_vocab_size_divisible_by=128 \
--seq_length=4096 \
--max_position_embeddings=4096 \
--position_embedding_type=rotary \
--glu_activation=swiglu \
--use_rms_norm \
--layernorm_epsilon=1e-5 \
--no_tie_embed_logits \
--no_bias_gelu_fusion \
--no_bias_dropout_fusion \
--hidden_dropout=0.0 \
--attention_dropout=0.0 \
--log_interval=1 \
--save_interval=2000 \
--eval_interval=100 \
--log_params_norm \
--log_timers_to_tensorboard \
--wandb_logger \
--wandb_project=llama-megatron-3nodes \
--wandb_id=6_fp16_grad1_mtl-initialization \
--wandb_api_key=fac46169cec8e164a47ed1c71199e3e8e9f02cc5 \
2>&1 |tee /home/xinghq/megatron-llm/logs/3nodes.txt
