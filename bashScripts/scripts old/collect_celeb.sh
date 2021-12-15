#!/bin/bash

python main.py --gpu_id="2" --experiment_id="progan_17s" --optional_run_id="paper_j" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --COMPUTE_STATISTICS_OFF --start=51 --end=53 --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --n_frames=80 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


python main.py --gpu_id="2" --experiment_id="progan_17s" --optional_run_id="paper_b" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --COMPUTE_STATISTICS_OFF --start=583 --end=86 --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --n_frames=80 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


python main.py --gpu_id="2" --experiment_id="progan_17s" --optional_run_id="paper_c" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --COMPUTE_STATISTICS_OFF --start=83 --end=56 --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --n_frames=80 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"

python main.py --gpu_id="2" --experiment_id="progan_17s" --optional_run_id="paper_m" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --COMPUTE_STATISTICS_OFF --start=53 --end=26 --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --n_frames=80 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


python main.py --gpu_id="2" --experiment_id="progan_17s" --optional_run_id="paper_n" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --COMPUTE_STATISTICS_OFF --start=53 --end=35 --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --n_frames=80 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"

python main.py --gpu_id="2" --experiment_id="progan_17s" --optional_run_id="paper_o" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --COMPUTE_STATISTICS_OFF --start=53 --end=43 --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --n_frames=80 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"

l
e
