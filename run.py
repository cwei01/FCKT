import os
import subprocess
import numpy as np
import sys

def run(command):
    subprocess.call(command, shell=True)

def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    parameter, result, time_cost = '', '', ''
    for line in iter(p.stdout.readline, b''):
        line = line.rstrip().decode('utf8')
        if 'args:' in line:
            parameter = line
        if 'final best performance:' in line:
            result = line
        if 'Experiment cost:' in line:
            time_cost = line
    return parameter, result, time_cost

def per_data():
    for random_train in [0.9]:
        for logit_threshold in [7.5]:
          for learning_rate in [2e-5]:
             for  weight_span in [1e-7]:
                    path = os.path.join('out/Res', "Nov29-%s-%s-%s-%s" % (random_train,logit_threshold,learning_rate,weight_span))
                    cmd = 'python -m absa.run_joint ' +' '+ \
                          ' --logit_threshold ' +  str(logit_threshold)+' '+ \
                          ' --learning_rate ' +  str(learning_rate)+' '+ \
                          ' --train_batch_size ' + str(16) + ' ' + \
                          ' --num_train_epochs 100 ' + ' '+ \
                          ' --weight_span '  +  str(weight_span)+ ' ' + \
                          ' --output_dir '  + str(path) + ' '+ \
                          ' --random_train ' + str(0.9) +' '
                    run(cmd)
                    sys.stdout.flush()
per_data()
