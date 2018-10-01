#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log

/media/dml/1TbyeAvailable/libraries/caffe/build/tools/caffe train -solver solver.prototxt -snapshot /media/dml/1TbyeAvailable/caffe-moon/models/visualization_models_pca/New_iter_590000.solverstate -gpu 1 2>&1 | tee $LOG

