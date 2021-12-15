#!/bin/bash


id="b"
start=2
end=4

python main.py --gpu_id="0" --start=${start} --end=${end}  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="e"
start=2
end=75

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"

id="f"
start=13
end=4

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="g"
start=59
end=17

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="h"
start=59
end=71

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="n"
start=22
end=4

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"

id="o"
start=22
end=17

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="r"
start=81
end=87

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="s"
start=81
end=89

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="u"
start=81
end=91

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"


id="z"
start=54
end=34

python main.py --gpu_id="0" --start=${start} --end=${end} --COMPUTE_STATISTICS_OFF  --experiment_id="progan_17s3" --optional_run_id="bad_search_${id}" --model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" --geodesic_training_steps=300 --hyperparam_disc_vs_mse=0.15 --hyperparam_disc_vs_vgg=0.15 --methods "linear" "mse" "mse_plus_disc" "vgg" "vgg_plus_disc"