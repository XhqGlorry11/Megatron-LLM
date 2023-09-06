import os
pid = list(set(os.popen('fuser -v /dev/nvidia*').read().split()))
pid_str = ''
for p in pid:
    pid_str += (' ' + p)
print (pid_str)