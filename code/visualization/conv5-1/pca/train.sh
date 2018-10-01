#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver solver.prototxt -weights=New_iter_80000.caffemodel -gpu 0 2>&1 | tee $LOG

