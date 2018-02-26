#!/bin/bash
cd /media/dml/1TbyeAvailable/caffe-moon/result/conv7/train
LOG=../log/dpfe_finetune_all_`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver ../solver/dpfe_all_solver.prototxt -weights=../model/pca/pca_iter_3000.caffemodel  2>&1  | tee $LOG
