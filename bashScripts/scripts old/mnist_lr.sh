#!/bin/bash

gpu_id=2

python main.py --experiment_id="mnist_hypers" --gpu_id=$gpu_id --model="../../progan/results_mnist/001-pgan-mnist-mnist_training-1gpu-fp32/network-final.pkl" \
--tfrecord_dir='../../MNISTgeodesics/dataset/tfrecords' \
--VIDEO_GENERATION_OFF \
--TRAIN_GEODESICS_OFF \
--n_statistics=5000 \
--dim_latent=128 \


# digits 
for geodesic_training_steps in 500 1000 2000 3000 4000 5000 10000 20000 50000
do
	#digits 
	for hyperparam_disc_vs_mse in 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.4 0.6 0.8 1.0 2.0 5.0 10.0  
	do
   
		python main.py --experiment_id="mnist_hypers" --gpu_id=$gpu_id --model="../../progan/results_mnist/001-pgan-mnist-mnist_training-1gpu-fp32/network-final.pkl" \
		--tfrecord_dir='../../MNISTgeodesics/dataset/tfrecords' \
		--VIDEO_GENERATION_OFF \
		--COMPUTE_STATISTICS_OFF \
		--dim_latent=128 \
		--n_pts_on_geodesic=24 \
		--geodesic_training_steps=3000 \
    	--hyperparam_disc_vs_mse=0.1 \
		--start=98 \
		--end=60 \
		--optional_run_id="${start}_${end}_init${init}_lr${}" \
		--methods "linear" "linear_in_sample" "disc" "mse" "mse_plus_disc"
 	done
done



# good seeds per digit
#0 13 33* 74 82* 100
#1 19*, 44*, 51, 64
#2 60*, 75*, 85, 88
#3 8, 26*, 43*, 67,  
#4 27, 37*, 56, 65*, 83, 93, 96
#5 35*, 77, 98*
#6 4, 12, 14*, 28*, 29, 55, 94, 
#7 7, 10*, 15, 66, 79*
#8 9, 24*, 86*, 
#9 36*, 59, 62*, 81, 
# all selected: #33 82 19 44 60 75 25 43 37 65 35 98 14 28 10 79 24 86 36 62


# Selected interpolations:

# 	11 - 9: 8 to 8 via 7
#	12 - 1: 6 to 7 via 8
#	12 - 3: 6 to 8 via 0
#	12 - 5: 6 to 1 artefacts
#	12 - 7: 6 to 7 art*
#	12 - 9: 6 to 8 via 0
#	12 - 10: 6 to 7 via 4 and 9
#	13 - 7
# 	13 - 10
# 	14 - 1







