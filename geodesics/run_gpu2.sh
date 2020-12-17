#!/bin/bash


# python ~/github/CelebAgeodesics/geodesics/main.py --startpoint_seed 200 --endpoint_seed 250 --subfolder_path "comparing_seeds/" --file_name "200_250"
# python ~/github/CelebAgeodesics/geodesics/main.py --startpoint_seed 201 --endpoint_seed 251 --subfolder_path "comparing_seeds/" --file_name "201_251"
# python ~/github/CelebAgeodesics/geodesics/main.py --startpoint_seed 202 --endpoint_seed 252 --subfolder_path "comparing_seeds/" --file_name "202_252"
# python ~/github/CelebAgeodesics/geodesics/main.py --startpoint_seed 203 --endpoint_seed 253 --subfolder_path "comparing_seeds/" --file_name "203_253"
# python ~/github/CelebAgeodesics/geodesics/main.py --startpoint_seed 204 --endpoint_seed 254 --subfolder_path "comparing_seeds/" --file_name "204_254"
# python ~/github/CelebAgeodesics/geodesics/main.py --startpoint_seed 205 --endpoint_seed 255 --subfolder_path "comparing_seeds/" --file_name "205_255"

# python ~/github/CelebAgeodesics/geodesics/main.py --hyper_critic_penalty 100 --subfolder_path "comparing_penalties/" --file_name "100"
# python ~/github/CelebAgeodesics/geodesics/main.py --hyper_critic_penalty 10 --subfolder_path "comparing_penalties/" --file_name "10"
# python ~/github/CelebAgeodesics/geodesics/main.py --hyper_critic_penalty 1 --subfolder_path "comparing_penalties/" --file_name "1"
# python ~/github/CelebAgeodesics/geodesics/main.py --hyper_critic_penalty 0.1 --subfolder_path "comparing_penalties/" --file_name "0_1"
# python ~/github/CelebAgeodesics/geodesics/main.py --hyper_critic_penalty 0.01 --subfolder_path "comparing_penalties/" --file_name "0_01"
# python ~/github/CelebAgeodesics/geodesics/main.py --hyper_critic_penalty 0.001 --subfolder_path "comparing_penalties/" --file_name "0_001"
# python ~/github/CelebAgeodesics/geodesics/main.py --hyper_critic_penalty 0.0001 --subfolder_path "comparing_penalties/" --file_name "0_0001"

# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_learning_rate 1 --subfolder_path "comparing_learning_rates/" --file_name "1"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_learning_rate 0.1 --subfolder_path "comparing_learning_rates/" --file_name "0_1"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_learning_rate 0.01 --subfolder_path "comparing_learning_rates/" --file_name "0_01"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_learning_rate 0.001 --subfolder_path "comparing_learning_rates/" --file_name "0_001"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_learning_rate 0.0001 --subfolder_path "comparing_learning_rates/" --file_name "0_0001"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_learning_rate 0.00001 --subfolder_path "comparing_learning_rates/" --file_name "0_00001"

# python ~/github/CelebAgeodesics/geodesics/main.py --coefficient_init 1 --subfolder_path "comparing_init_range/" --file_name "1"
# python ~/github/CelebAgeodesics/geodesics/main.py --coefficient_init 0.1 --subfolder_path "comparing_init_range/" --file_name "0_1"
# python ~/github/CelebAgeodesics/geodesics/main.py --coefficient_init 0.01 --subfolder_path "comparing_init_range/" --file_name "0_01"
# python ~/github/CelebAgeodesics/geodesics/main.py --coefficient_init 0.001 --subfolder_path "comparing_init_range/" --file_name "0_001"
# python ~/github/CelebAgeodesics/geodesics/main.py --coefficient_init 0.0001 --subfolder_path "comparing_init_range/" --file_name "0_0001"
# python ~/github/CelebAgeodesics/geodesics/main.py --coefficient_init 0.00001 --subfolder_path "comparing_init_range/" --file_name "0_00001"

# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_training_steps 10 --subfolder_path "comparing_iteration_steps/" --file_name "10"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_training_steps 100 --subfolder_path "comparing_iteration_steps/" --file_name "100"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_training_steps 1000 --subfolder_path "comparing_iteration_steps/" --file_name "1000"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_training_steps 5000 --subfolder_path "comparing_iteration_steps/" --file_name "5000"
# python ~/github/CelebAgeodesics/geodesics/main.py --geodesic_training_steps 20000 --subfolder_path "comparing_iteration_steps/" --file_name "20000"


# COMPARE SEEDS

gpu_id=2
subpath="comparing_seeds/"
nbr_iterations=10000


start=200
end=250
python ~/github/CelebAgeodesics/geodesics/main.py --gpu_id $gpu_id --geodesic_training_steps $nbr_iterations \
	--subfolder_path $subpath \
	--startpoint_seed $start --endpoint_seed $end --file_name "${start}_${end}"
	
start=204
end=254
python ~/github/CelebAgeodesics/geodesics/main.py --gpu_id $gpu_id --geodesic_training_steps $nbr_iterations \
	--subfolder_path $subpath \
	--startpoint_seed $start --endpoint_seed $end --file_name "${start}_${end}"
	
start=205
end=255
python ~/github/CelebAgeodesics/geodesics/main.py --gpu_id $gpu_id --geodesic_training_steps $nbr_iterations \
	--subfolder_path $subpath \
	--startpoint_seed $start --endpoint_seed $end --file_name "${start}_${end}"
	
start=206
end=251
python ~/github/CelebAgeodesics/geodesics/main.py --gpu_id $gpu_id --geodesic_training_steps $nbr_iterations \
	--subfolder_path $subpath \
	--startpoint_seed $start --endpoint_seed $end --file_name "${start}_${end}"

start=203
end=256
python ~/github/CelebAgeodesics/geodesics/main.py --gpu_id $gpu_id --geodesic_training_steps $nbr_iterations \
	--subfolder_path $subpath \
	--startpoint_seed $start --endpoint_seed $end --file_name "${start}_${end}"


