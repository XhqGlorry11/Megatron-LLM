# -*- coding: utf-8 -*-
# @Time : 2023/6/20 3:14
# @Author : Linyi Zuo @ ThunderSoft
# @Project : gpt-neox-aws
# @File : sample_start.py.py
# @Software: PyCharm
import os
import json
import socket

if __name__ == "__main__":
  hosts = json.loads(os.environ['SM_HOSTS'])
  current_host = os.environ['SM_CURRENT_HOST']
  host_rank = int(hosts.index(current_host))

  master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
  master_addr = socket.gethostbyname(master)

  os.environ['NODE_INDEX'] = str(host_rank)
  os.environ['SM_MASTER'] = str(master)
  os.environ['SM_MASTER_ADDR'] = str(master_addr)
  os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
  print('='*50, 'start user scripts', flush=True)

  os.system("chmod +x ./s5cmd")
  os.system("chmod +x ./torch_launch.sh")
  os.system("/bin/bash -c ./torch_launch.sh")