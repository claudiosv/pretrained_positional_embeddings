#!/bin/bash

# export 
#0,1,3,4,5
CUDA_VISIBLE_DEVICES=4 python baseline_perceiver_cifar.py;
CUDA_VISIBLE_DEVICES=5 python run_perceiver_w_MAED_encoder.py;
# python run_perceiver_w_MAED_encoder.py