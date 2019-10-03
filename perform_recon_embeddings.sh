#!/bin/bash

#SBATCH --job-name=embeddingsNC
#SBATCH --output=embeddingsNC_%A_%a.out
#SBATCH --error=embeddingsNC_%A_%a.err
#SBATCH --array=0-599
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G


e=5

datasets=({cora_ml,citeseer,pubmed})
dims=(5 10 25 50)
seeds=({0..9})
ks=(02 03 04 05 06)

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_ks=${#ks[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_ks % num_seeds ))
k_id=$((SLURM_ARRAY_TASK_ID % (num_ks) ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
k=${ks[k_id]}

data_dir=../aheat/datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv
features=${data_dir}/feats.csv
embedding_dir=embeddings/${dataset}/recon_experiment

embedding_dir=$(printf "${embedding_dir}/k=${k}/seed=%03d/dim=%03d/" ${seed} ${dim})
echo ${embedding_dir}

if [ ! -f ${embedding_dir}"mu.csv" ]
then 
	module purge
	module load bluebear
	module load apps/python3/3.5.2
	module load apps/keras/2.0.8-python-3.5.2

	args=$(echo --edgelist ${edgelist} --features ${features} \
	--embedding ${embedding_dir} --seed ${seed} --dim ${dim} \
	"-k" ${k})

	# echo $args

	python embed.py ${args}
fi