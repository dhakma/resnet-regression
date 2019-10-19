#!/bin/bash
#nohup python resnet.py  --data_dir 'data/sketch-gen-holes' --train 2>&1 &
conda activate pytorch
#nohup python resnet.py  --data_dir 'data/multi-sketch-gen-3' --train > train.log 2>&1 &
#nohup python classification.py  --data_dir 'data/classification-data' --train > classifier_train.log 2>&1 &
nohup python -m visdom.server > visdom.log 2>&1 &
nohup python resnet.py  --data_dir 'data/path-v11' --train --resnet_type 'next101' > path_train.log 2>&1 &
nohup python resnet.py  --data_dir 'data/area-v10' --train --resnet_type 'next101' > area_train.log 2>&1 &
#nohup python resnet.py  --data_dir 'data/wall-v6' --train > wall_train.log 2>&1 &
tail -f path_train.log
#tail -f area_train.log
