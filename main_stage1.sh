#!/bin/sh
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=0 python main_stage1.py --dataset multi --source real --target clipart --net resnet34 --save_check --num 1 --save_interval 1000 --exp_variation s1_s+t
