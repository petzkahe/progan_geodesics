#!/bin/bash

geodesic_training_steps=100
hyperparam_disc_vs_mse=0.6
polynomial_degree=5

for start in 33 82 19 44 60 75 25 43 37 65 35 98 14 28 10 79 24 86 36 62
do
  for end in 33 82 19 44 60 75 25 43 37 65 35 98 14 28 10 79 24 86 36 62
  do
    if (("$start"<"$end")); then
    

      python main.py --experiment_id="mnist_paper" --gpu_id="0" --model="../../progan/results_mnist/001-pgan-mnist-mnist_training-1gpu-fp32/network-final.pkl" \
		    --tfrecord_dir='../../MNISTgeodesics/dataset/tfrecords' \
		    --VIDEO_GENERATION_OFF \
		    --COMPUTE_STATISTICS_OFF \
		    --dim_latent=128 \
		    --n_pts_on_geodesic=24 \
		    --geodesic_training_steps=${geodesic_training_steps} \
        --hyperparam_disc_vs_mse=${hyperparam_disc_vs_mse} \
        --polynomial_degree=${polynomial_degree} \
		    --start=${start} \
		    --end=${end} \
		    --optional_run_id="${start}_${end}_${geodesic_training_steps}_${hyperparam_disc_vs_mse}_${polynomial_degree}" \
		    --methods "linear" "mse" "mse_plus_disc"
    fi
  done
done 
        
        