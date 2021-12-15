 #!/bin/bash

# good seeds: [[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]

start=477
end=56
gpu_id=3
experiment_id="disc_test_low_lr"


#model_path="../../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/"
model_path="../../progan/results/012-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32"

for i in  25 30 35 40 45 50 60 70 80 90 100 120 140 160 180 200 240 280 320 365 366 367 368 369 370  
do
	if [ $i -lt 100 ]
	then
		if [ $i -lt 10 ]
		then 
			i_text="00${i}"
		else
			i_text="0${i}"
		fi
	else
		i_text="${i}"
	fi
	python main.py --TRAIN_GEODESICS_OFF --VIDEO_GENERATION_OFF --n_statistics=1000 --experiment_id=$experiment_id --optional_run_id="disc_at_${i_text}" --gpu_id=$gpu_id --model="${model_path}/network-snapshot-012${i_text}.pkl" --start=$start --end=$end --hyperparam_disc_vs_vgg=0.01 --methods "disc" "vgg_plus_disc"
done


#for i in 10 12 14 16 18 20 30 40 50 60 70 80 90
#do 
#   python main.py --COMPUTE_STATISTICS_OFF --TRAIN_GEODESICS_OFF --VIDEO_GENERATION_OFF --experiment_id="disc_test" --optional_run_id="disc_at_${i}" --gpu_id=3 --model="../../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-0120${i}.pkl" --start=477 --end=56 --hyperparam_disc_vs_mse=0.01 --VIDEO_GENERATION_OFF --methods "disc" "vgg_plus_disc"
# done


# for i in 100 120 139 159 179 199 219 239 260 280 300 340 379 419 459 499
# do 
#   python main.py --COMPUTE_STATISTICS_OFF --TRAIN_GEODESICS_OFF --VIDEO_GENERATION_OFF --experiment_id="disc_test" --optional_run_id="disc_at_${i}" --gpu_id=3 --model="../../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012${i}.pkl" --start=477 --end=56 --hyperparam_disc_vs_vgg=0.01 --VIDEO_GENERATION_OFF --methods "disc" "vgg_plus_disc"
# done

