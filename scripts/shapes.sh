#!/bin/bash

shapes3d=true
num_support_sets=5
num_timesteps=14
batch_size=128
max_iter=100000
tensorboard=true

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python latent_flow/train_vae_scratch.py $tb \
                --num-support-sets=${num_support_sets} \
                --num-timesteps=${num_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100 \
                --shapes3d=${shapes3d} 
