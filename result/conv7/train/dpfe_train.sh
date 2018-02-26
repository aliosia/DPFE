#!/bin/bash
cd /media/dml/1TbyeAvailable/caffe-moon/result/conv7/train
LOG=../log/dpfe_finetune_[20-39-31-6-7]`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver ../solver/dpfe_solver.prototxt -weights=../model/[20-39-31-6-7]/simple_finetune_[20-39-31-6-7]_iter_3000.caffemodel  2>&1  | tee $LOG