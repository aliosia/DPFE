#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver solver_rms.prototxt -weights=enc-pca-conv4-2_dec-alexnet.caffemodel  -gpu 0 2>&1 | tee $LOG
