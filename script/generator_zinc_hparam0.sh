#!/bin/bash

python train_generator.py \
--emb_size 512 \
--disable_valencemask \
--dataset_name zinc \
--tag zinc_hparam0
