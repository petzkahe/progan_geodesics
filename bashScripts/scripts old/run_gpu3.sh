#!/bin/bash

# good seeds: [[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]

gpu_id=3
experiment_id="my_new_experiment"
optional_run_id="test"
nbr_iterations=1000



start=477
end=56

python main.py \
  --gpu_id $gpu_id \
  --geodesic_training_steps $nbr_iterations \
	--experiment_id $experiment_id \
   --optional_run_id $optional_run_id
	--start $start --end $end \

