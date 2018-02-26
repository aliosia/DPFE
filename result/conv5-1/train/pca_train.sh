#!/bin/bash
cd /media/dml/1TbyeAvailable/caffe-moon/result/conv5-1/train
LOG=../log/pca_train_`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver ../solver/pca_solver.prototxt -weights=../model/pca/pca_conv5-1.caffemodel 2>&1 | tee $LOG

