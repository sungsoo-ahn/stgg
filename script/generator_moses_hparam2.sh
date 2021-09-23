#!/bin/bash

python train_generator.py \
--lr 5e-4 \
--num_layers 3 \
--input_dropout 0.0 \
--randomize \
--dataset_name moses \
--check_sample_every_n_epoch 5 \
--num_samples 30000 \
--eval_moses \
--tag generator_moses_hparam2