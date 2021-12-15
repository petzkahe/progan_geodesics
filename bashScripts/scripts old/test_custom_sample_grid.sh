#!/bin/bash

# CelebA-HQ
#python main.py --gpu_id="2" --experiment_id="progan_17s" \
#			--optional_run_id="paper_j" \
#			--model="../../progan/results/017-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012071.pkl" \
#			--COMPUTE_STATISTICS_OFF \
#			--TRAIN_GEODESICS_OFF \
#			--VIDEO_GENERATION_OFF \
#			--CUSTOM_SAMPLE_GRIDS_ON

# LSUN Car
python main.py --gpu_id="2" --experiment_id="LSUN" \
			--optional_run_id="test_stats" \
			--model="../../LSUNCARgeodesics/models/karras2018iclr-lsun-car-256x256.pkl" \
			--COMPUTE_STATISTICS_OFF \
			--TRAIN_GEODESICS_OFF \
			--VIDEO_GENERATION_OFF \
			--CUSTOM_SAMPLE_GRIDS_ON

#			--model="../../progan/results_lsuncar/001-pgan-lsun-car-lsun-car-from-pretrained-v1-1gpu-fp32/network-snapshot-007871.pkl" \
