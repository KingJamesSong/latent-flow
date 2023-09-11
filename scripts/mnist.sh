#!/bin/bash

num_support_sets=3
num_timesteps=16
batch_size=128
max_iter=100000
tensorboard=true

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python train_vae_scratch.py $tb \
                --num-support-sets=${num_support_sets} \
                --num-timesteps=${num_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100 \
               
