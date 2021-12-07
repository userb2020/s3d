#!/bin/sh
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=0 python main_stage2.py --dataset multi --source real --target clipart --net resnet34 --save_check --num 1 --save_interval 1000 --s1_exp_variation s1_s+t --exp_variation s2_s3d --pseudo_interval 100 --kd_lambda 8 --sty_layer layer4
