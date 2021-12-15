


for start in 1 2 3 
do
  for end in 1 2 3
    do
      if (("${start}"<"${end}")); then
        python main.py --experiment_id="bash_test" --gpu_id="0"  --model="../progan/results/008-pgan-celebahq-continue_training_discriminator-v1-1gpu-fp32/network-snapshot-012500.pkl" --start=$start --end=$end --optional_run_id="${start}_${end}" --hyper_sqDiff=0.01 --hyper_vgg=0.01 --n_training_steps=300 
      fi
    done
done