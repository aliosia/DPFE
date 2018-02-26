
# Finetunning all simple models for all chosen attribute sets and all intermediate layers.
# Results are saved in 'intermediate_layer/model/attribute_set/simple_finetune_itername.caffemodel

import os
import subprocess


def recreate_filter_file(filter_path, filt_str):
    with open(filter_path, 'r') as file:
        data = file.readlines()

    for i,line in enumerate(data):
        if 'indices =' in line:
            newline = line[0:line.find('=')+2] + filt_str + '\n'
            data[i] = newline
            
    with open(filter_path, 'w') as file:
        file.writelines( data )





def recreate_train_sh(path, filt_str):
    with open(path, 'r') as file:
        data = file.readlines()
        
    for i,line in enumerate(data):
        if '`date' in line:
            newline = 'LOG=../log/simple_finetune_'+ filt_str + '`date +%Y-%m-%d-%H-%M-%S`.log' + '\n'
            data[i] = newline
        if '-weights' in line:
            newline = '/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver ../solver/simple_solver.prototxt -weights=../model/pca/pca_iter_3000.caffemodel 2>&1 | tee $LOG'
            data[i] = newline
    with open(path, 'w') as file:
        file.writelines( data )





def recreate_solver(path, filt_str):
    with open(path, 'r') as file:
        data = file.readlines()
        
    for i,line in enumerate(data):
        if 'net:' in line:
            newline = 'net: "../prototxt/train_simple_' + filt_str + '.prototxt"' + '\n'
            data[i] = newline
        if 'snapshot_prefix' in line:
            newline = 'snapshot_prefix: "../model/' + filt_str + '/simple_finetune_' + filt_str + '"' + '\n'
            data[i] = newline
            
    with open(path, 'w') as file:
        file.writelines( data )


def create_model_folder(path, filt_str):
    new_dir = os.path.join(path, filt_str)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)




layer_names = ['conv4-2', 'conv5-1', 'conv6-1', 'conv7']
filt_strs = ['[20]', '[20,39]', '[20,39,31]', '[20,39,31,6]', '[20,39,31,6,7]']
filt_names = ['[20]', '[20-39]', '[20-39-31]', '[20-39-31-6]', '[20-39-31-6-7]']



filter_path = '/media/dml/1TbyeAvailable/libraries/caffe/python/filter.py'
base_path = '/media/dml/1TbyeAvailable/caffe-moon/result/'

for layer_name in layer_names:
    for i,filt_str in enumerate(filt_strs):
        train_path = base_path + layer_name + '/train/simple_train.sh'
        solver_path = base_path + layer_name + '/solver/simple_solver.prototxt'
        create_model_folder(base_path + layer_name +'/model/', filt_names[i])
        recreate_filter_file(filter_path, filt_str)
        recreate_train_sh(train_path, filt_names[i])
        recreate_solver(solver_path, filt_names[i])
        subprocess.call(train_path, shell=True)






