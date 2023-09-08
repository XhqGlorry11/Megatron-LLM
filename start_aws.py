import os.path
import time

# from sagemaker.estimator import Estimator
from sagemaker.huggingface import HuggingFace
import sagemaker
import boto3
from sagemaker import get_execution_role
from datetime import datetime

from sagemaker.inputs import FileSystemInput

# 获取当前时间
now = datetime.now()

# 转换为需要的格式: 年-月-日-时-分-秒
formatted_now_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

sess = sagemaker.Session()
# role = get_execution_role()
role = 'arn:aws:iam::868681190882:role/sagemake'
sagemaker_default_bucket = sess.default_bucket()

# account = sess.boto_session.client("sts").get_caller_identity()["Account"]
region = sess.boto_session.region_name

# repo_name = 'tsgptx'

# image_uri = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(account, region, repo_name)
# image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04'
# image_uri = '868681190882.dkr.ecr.us-west-2.amazonaws.com/mtlm_train_aws:latest'
image_uri = '868681190882.dkr.ecr.us-west-2.amazonaws.com/mtlm-torch-training:2.0.0-transf4.31.0-py310-cu118-ubuntu20.04-sagemaker-dev'
## set train_data_path to your training dataset path in s3
# train_data_path = 's3://thundercomm-data-no69y/pretrain/350M/ts_data_tstokv2_20020717/'
# train_data_path = 's3://thundercomm-data-no69y/pretrain/350M/llama2_merged_mini/'
# train_data_path = 's3://thundercomm-data-no69y/gpt125M/train_data_mini/'
train_data_path = 's3://thundercomm-data-no69y/pretrain/continue_pretrain_250B'  # 2023年08月20日上载完成

model_data_path = 's3://thundercomm-models-4ynqb/pretrain/pretrained/llama-2-7b-ex-megatron-tp4-pp2/'
# model_data_path = 's3://thundercomm-models-4ynqb/pretrain/pretrained/llama-2-7b-ex-megatron-tp4-pp2-continue-test/'

save_path = f's3://ts-checkpoints-us-west-2/language/pretrain/'

inputs = {'train': train_data_path}
print(inputs)

node_number = 32

instance_type = 'ml.p4de.24xlarge'
base_job_name = f"yu-m-west2-ll-zuoly-c-1"
MODEL_TYPE='lan'
FRAMEWORK='mlm'
AREA='west2'
MODEL_NAME='llama2'
USER_NAME='zuoly'
TRAIN_TYPE='cp'
TRAIN_NUM='1'
s3_save_path = os.path.join(save_path, f'{MODEL_TYPE}-{FRAMEWORK}-{AREA}-{MODEL_NAME}-{USER_NAME}-{TRAIN_TYPE}-{TRAIN_NUM}')
environment = {
  'CUDA_DEVICE_MAX_CONNECTIONS': '1',
  'WANDB_API_KEY': 'bd93c2be40e37136af161080a78796dc6e776554',
  'MODEL_S3_BUCKET': sagemaker_default_bucket,  # The bucket to store pretrained model and fine-tune model
  'S3_DATA_PATH': os.path.join(train_data_path, '*'),
  'S3_MODEL_PATH': os.path.join(model_data_path, '*'),
  'S3_SAVE_PATH': s3_save_path + '/',
  'NODE_NUMBER': str(node_number),
  # 'CUDA_LAUNCH_BLOCKING': '1',
  'FI_PROVIDER': 'efa',
  'NCCL_PROTO': 'simple',
  'FI_EFA_USE_DEVICE_RDMA': '1',
  'NCCL_DEBUG': 'INFO',
  'JOB_NAME': base_job_name,
  # 'NCCL_ASYNC_ERROR_HANDLING': '1',
  # 'NCCL_BLOCKING_WAIT': '1',
  # 'CUDA_HOME': '/usr/local/cuda-11.8',
  # 'LD_LIBRARY_PATH': "/usr/local/cuda-11.8/lib64/:$LD_LIBRARY_PATH",
}

print(environment)

# estimator = Estimator(role=role,
#                       #   entry_point='sagemake_env_setup.py',
#                       entry_point='sample_start.py',
#                       source_dir='./',
#                       base_job_name=base_job_name,
#                       instance_count=node_number,
#                       instance_type=instance_type,
#                       image_uri=image_uri,
#                       environment=environment,
#                       disable_profiler=True,
#                       max_run=5 * 24 * 60 * 60,
#                       keep_alive_period_in_seconds=30 * 60,
#                       tags=[{
#                         'Key': 'map-migrated',
#                         'Value': 'migHMRKOJA44X',
#                       }],
#                       debugger_hook_config=False)
print(instance_type, '*', node_number)

# Specify FSx Lustre file system id.
file_system_id = "fs-0ce2cdef6b66b6740"  # Change to your Fsx FS id
# Specify directory path for input data on the file system.
# You need to provide normalized and absolute path below.
file_system_directory_path = '/adjglbev/continue_pretrain_250B'  # Change to your Fsx Mount name which is given in FSx FS details
# Specify the access mode of the mount of the directory associated with the file system.
file_system_access_mode = 'rw'
# Specify your file system type.
file_system_type = 'FSxLustre'
fsx_fs = FileSystemInput(file_system_id=file_system_id,
                         file_system_type=file_system_type,
                         directory_path=file_system_directory_path,
                         file_system_access_mode=file_system_access_mode)
fsx_channels = {'fsx': fsx_fs}

estimator = HuggingFace(
  image_uri=image_uri,
  entry_point='entry.py',  # deepspeed launcher script
  source_dir='.',  # directory which includes all the files needed for training
  instance_type=instance_type,  # instances type used for the training job
  instance_count=node_number,  # the number of instances used for training
  base_job_name=base_job_name,  # the name of the training job
  role=role,  # Iam role used in training job to access AWS ressources, e.g. S3
  # volume_size          = 600,               # the size of the EBS volume in GB
  # transformers_version='4.17',  # the transformers version used in the training job
  # pytorch_version='1.13',  # the pytorch_version version used in the training job
  py_version='py310',  # the python version used in the training job
  keep_alive_period_in_seconds=3600,
  environment=environment,
  debugger_hook_config=False,
  tags=[{
    'Key': 'map-migrated',
    'Value': 'migHMRKOJA44X',
  }],
  max_run=20 * 24 * 60 * 60,

  subnets=['subnet-0cd5709c65b8529fe'],  # Should be same vpc with FSx, best to use same subnet with FSx
  security_group_ids=['sg-0ccd5f2037bb946bc'],  # Needed when use FSx

  # checkpoint_local_path = "/workspace/output/",
  checkpoint_s3_uri=s3_save_path,
  # will create /opt/ml/checkpoints for this uri
)
estimator.fit(wait=True, inputs=fsx_channels)
# estimator.fit(inputs, wait=True)
