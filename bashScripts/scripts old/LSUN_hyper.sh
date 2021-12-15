start=12
end=30

python main.py --gpu_id="3" --model="../../LSUNBEDgeodesics/models/karras2018iclr-lsun-bedroom-256x256.pkl" --experiment_id="bedrooms" --optional_run_id="paper_${start}_${end}" --COMPUTE_STATISTICS_OFF --VIDEO_GENERATION_OFF --tfrecord_dir="../../LSUNBEDgeodesics/dataset/tfrecords" --start=${start} --end=${end} --methods "linear" "vgg" "vgg_plus_disc" --polynomial_degree=3 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --geodesic_training_steps=200



start=17
end=20

python main.py --gpu_id="3" --model="../../LSUNBEDgeodesics/models/karras2018iclr-lsun-bedroom-256x256.pkl" --experiment_id="bedrooms" --optional_run_id="paper_${start}_${end}" --COMPUTE_STATISTICS_OFF --VIDEO_GENERATION_OFF --tfrecord_dir="../../LSUNBEDgeodesics/dataset/tfrecords" --start=${start} --end=${end} --methods "linear" "vgg" "vgg_plus_disc" --polynomial_degree=3 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --geodesic_training_steps=200



start=20
end=73

python main.py --gpu_id="3" --model="../../LSUNBEDgeodesics/models/karras2018iclr-lsun-bedroom-256x256.pkl" --experiment_id="bedrooms" --optional_run_id="paper_${start}_${end}" --COMPUTE_STATISTICS_OFF --VIDEO_GENERATION_OFF --tfrecord_dir="../../LSUNBEDgeodesics/dataset/tfrecords" --start=${start} --end=${end} --methods "linear" "vgg" "vgg_plus_disc" --polynomial_degree=3 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --geodesic_training_steps=200




start=20
end=93

python main.py --gpu_id="3" --model="../../LSUNBEDgeodesics/models/karras2018iclr-lsun-bedroom-256x256.pkl" --experiment_id="bedrooms" --optional_run_id="paper_${start}_${end}" --COMPUTE_STATISTICS_OFF --VIDEO_GENERATION_OFF --tfrecord_dir="../../LSUNBEDgeodesics/dataset/tfrecords" --start=${start} --end=${end} --methods "linear" "vgg" "vgg_plus_disc" --polynomial_degree=3 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --geodesic_training_steps=200



start=30
end=57


python main.py --gpu_id="3" --model="../../LSUNBEDgeodesics/models/karras2018iclr-lsun-bedroom-256x256.pkl" --experiment_id="bedrooms" --optional_run_id="paper_${start}_${end}" --COMPUTE_STATISTICS_OFF --VIDEO_GENERATION_OFF --tfrecord_dir="../../LSUNBEDgeodesics/dataset/tfrecords" --start=${start} --end=${end} --methods "linear" "vgg" "vgg_plus_disc" --polynomial_degree=3 --hyperparam_disc_vs_mse=0.2 --hyperparam_disc_vs_vgg=0.2 --geodesic_training_steps=200

