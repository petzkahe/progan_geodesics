for start in 12 16 17 20 29 30 32 37 45 57 60 64 67 73 79 85 86 87 93
do
  for end in 12 16 17 20 29 30 32 37 45 57 60 64 67 73 79 85 86 87 93
  do
    if (("${start}"<"${end}")); then
      python main.py --gpu_id="3" --model="../../progan/results_lsunbedroom/000-pgan-lsun-bedroom-lsun-bedroom-from-pretrained-v1-1gpu-fp32/network-snapshot-007871.pkl" --experiment_id="bedrooms_71" --optional_run_id="original_${start}_${end}" --COMPUTE_STATISTICS_OFF --VIDEO_GENERATION_OFF --tfrecord_dir="../../LSUNBEDgeodesics/dataset/tfrecords" --start=${start} --end=${end} --methods "linear" "disc" "vgg" "vgg_plus_disc" --polynomial_degree=5
    fi
  done
done
