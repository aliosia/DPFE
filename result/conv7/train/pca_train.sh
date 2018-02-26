#!/bin/bash
cd /media/dml/1TbyeAvailable/caffe-moon/result/conv7/train
LOG=../log/pca_train-`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver ../solver/pca_solver.prototxt -weights=../model/pca/pca_conv7.caffemodel 2>&1 | tee $LOG

