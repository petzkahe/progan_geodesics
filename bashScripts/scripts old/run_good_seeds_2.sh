#!/bin/bash
# COMPARE SEEDS
# good seeds: [[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]

#python main.py --experiment_id="progan_20s" --gpu_id=0 --model="../../progan/results/020-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012002.pkl" --TRAIN_GEODESICS_OFF --VIDEO_GENERATION_OFF


for i in 477 56 83 887 583 391 86 340 341 415 
#for i in 101 102 103 104 105 106 107 108 109 110
do
  for j in 477 56 83 887 583 391 86 340 341 415 
#   for j in 101 102 103 104 105 106 107 108 109 110
    do
      if (("$i">"$j")); then
        start=$i
        end=$j
        python main.py --experiment_id="progan_17s" --gpu_id=2  --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --start=$start --end=$end --optional_run_id="few_steps_${start}_${end}_71" --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --geodesic_training_steps=100 --COMPUTE_STATISTICS_OFF --methods "linear" "vgg_plus_disc"
      fi
    done
done