#!/bin/bash

python train_generator.py \
--dataset_name moses \
--eval_moses \
--batch_size 256 \
--tag moses_largebatch