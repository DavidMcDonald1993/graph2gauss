#!/bin/bash

#SBATCH --job-name=embeddingsLPSynthetic
#SBATCH --output=embeddingsLPSynthetic_%A_%a.out
#SBATCH --error=embeddingsLPSynthetic_%A_%a.err
#SBATCH --array=0-1499
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G

scales=(False True)
datasets=({00..29})
dims=(2 5 10 25 50)
seeds=(0)
ks=(02 03 04 05 06)

num_scales=${#scales[@]}
num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_ks=${#ks[@]}

scale_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds * num_dims * num_datasets) % num_scales))
dataset_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_ks % num_seeds ))
k_id=$((SLURM_ARRAY_TASK_ID % (num_ks) ))

scale=${scales[$scale_id]}
dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
k=${ks[k_id]}

data_dir=../HEDNet/datasets/synthetic_scale_free/${dataset}
edgelist=$(printf ../HEDNet/edgelists/synthetic_scale_free/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
embedding_dir=embeddings/synthetic_scale_free/${dataset}/lp_experiment

embedding_dir=$(printf "${embedding_dir}/scale=${scale}/k=${k}/seed=%03d/dim=%03d/" ${seed} ${dim})

if [ ! -f ${embedding_dir}"mu.csv" ]
then 
	module purge
	module load bluebear
	module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6

	args=$(echo --edgelist ${edgelist} \
	--embedding ${embedding_dir} --seed ${seed} --dim ${dim} \
	"-k" ${k} "--scale" ${scale})

	python embed.py ${args}
fi