 #!/bin/bash

# good seeds: [[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]

start=477
end=56
gpu_id=3
experiment_id="progan_compare_17_18_19"

#model_path="../../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/"
model_path="../../progan/results"

printf "before loop"
for model_id in "017" "018" "019"
do
	printf "first loop"
	for i in  1 10 50 100 275  
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
		python main.py --TRAIN_GEODESICS_OFF --VIDEO_GENERATION_OFF --n_statistics=3000 --experiment_id=$experiment_id --optional_run_id="progan_${model_id}_at_${i_text}" --gpu_id=$gpu_id --model="${model_path}/${model_id}-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012${i_text}.pkl" --start=$start --end=$end
	done
done

#   python main.py --COMPUTE_STATISTICS_OFF --TRAIN_GEODESICS_OFF --VIDEO_GENERATION_OFF --experiment_id="disc_test" --optional_run_id="disc_at_${i}" --gpu_id=3 --model="../../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-0120${i}.pkl" --start=477 --end=56 --hyperparam_disc_vs_mse=0.01 --VIDEO_GENERATION_OFF --methods "disc" "vgg_plus_disc"

