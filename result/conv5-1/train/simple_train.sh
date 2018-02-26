#!/bin/bash
cd /media/dml/1TbyeAvailable/caffe-moon/result/conv5-1/train
LOG=../log/simple_finetune_[20-39-31-6-7]`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver ../solver/simple_solver.prototxt -weights=../model/pca/pca_iter_3000.caffemodel 2>&1 | tee $LOG