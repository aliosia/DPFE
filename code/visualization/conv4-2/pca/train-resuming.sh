#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver solver_rms.prototxt -snapshot trained-model/rms_iter_390000.solverstate -gpu 1 2>&1 | tee $LOG

