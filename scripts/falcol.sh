#!/bin/bash

num_support_sets=7
num_timesteps=10
batch_size=128
max_iter=100000
tensorboard=true

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python latent_flow/train_falcol_isaac.py $tb \
                --num-support-sets=${num_support_sets} \
                --num-timesteps=${num_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100 \
