#!/bin/sh

script_name=$1
job_name=$2
prob=$3
alpha=$4
ver=$5

sbatch --partition=spgpu --gpus=1 --cpus-per-gpu=8 --nodes=1 --mem-per-cpu=11500m --time=00-3:00:00 --job-name="${job_name}_${ver}" --output="logs/${job_name}_${ver}.out" $script_name $prob $alpha $ver
