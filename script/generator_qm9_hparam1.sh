#!/bin/bash

python train_generator.py \
--emb_size 256 \
--randomize \
--disable_valencemask \
--dataset_name qm9 \
--tag qm9_hparam0