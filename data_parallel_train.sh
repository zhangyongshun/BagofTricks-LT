#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2

CUDA_VISIBLE_DEVICES=$GPUS $PYTHON main/train.py --cfg $CONFIG ${@:3}
