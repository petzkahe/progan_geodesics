#!/bin/bash
# COMPARE SEEDS
# good seeds: [[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]

#python main.py --experiment_id="good_seeds_500" --gpu_id=3 --model="../../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012500.pkl" --TRAIN_GEODESICS_OFF --VIDEO_GENERATION_OFF


for i in 477 56 83 887 583 391 86 340 341 415 
do
  for j in 477 56 83 887 583 391 86 340 341 415 
    do
      if (("$i">"$j")); then
        start=$i
        end=$j
        python main.py --experiment_id="good_seeds_500" --gpu_id=3  --model="../../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012500.pkl" --start=$start --end=$end --optional_run_id="${start}_${end}_small_hyper" --hyperparam_disc_vs_mse=0.01 --hyperparam_disc_vs_vgg=0.01 --COMPUTE_STATISTICS_OFF
      fi
    done
done