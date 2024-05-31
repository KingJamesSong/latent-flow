pool="VAE_MNIST_VIS"
eps=1
shift_steps=16
shift_leap=1
# =====================

declare -a EXPERIMENTS=("experiments/wip/VAE_MNIST-scratch-K3-D16-LearnGammas-eps0.0_8.1")

python sample_vae.py --num-samples 10 --pool "VAE_MNIST_VIS" -g "VAE_MNIST" \

wait

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space_vae.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap} 
                                
done
