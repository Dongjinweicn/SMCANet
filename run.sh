#!/bin/sh

 CUDA_VISIBLE_DEVICES=0 python -u main.py --model SMCAnet --eval --dataset ./input --sequence test_3modalities --resume ./pth/checkpoint0best6.pth --device cuda

