
# As the main model, the initial pca models use MSE loss and does not contain Batch Normalization layer
# So we should finetune them with Batch Normalization and Cross Entropy loss (This is because our theory is based on cross entropy. The results are similar even if we dont change the loss function.)


import os
import subprocess

layer_names = ['conv4-2', 'conv5-1', 'conv6-1', 'conv7']

base_path = '/media/dml/1TbyeAvailable/caffe-moon/result/'

for layer_name in layer_names:
    train_path = base_path + layer_name + '/train/pca_train.sh'
    subprocess.call(train_path, shell=True)


# The result of this finetunning are models (../models/pca/pca_iter_3000.caffemodel) finetuned with cross entropy loss which contain batch normalization layer on top of their middle layer due to variance stability. 



