#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver solver.prototxt -weights=_iter_210000.caffemodel -gpu 1 2>&1 | tee $LOG

