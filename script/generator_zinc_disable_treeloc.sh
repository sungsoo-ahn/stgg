#!/bin/bash

python train_generator.py \
--dataset_name zinc \
--num_layers 3 \
--disable_valencemask \
--disable_treeloc \
--tag zinc_valencemaskoff_treelocoff