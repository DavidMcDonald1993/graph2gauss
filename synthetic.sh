#!/bin/bash

#SBATCH --job-name=G2GembeddingsSYNTHETIC
#SBATCH --output=G2GembeddingsSYNTHETIC_%A_%a.out
#SBATCH --error=G2GembeddingsSYNTHETIC_%A_%a.err
#SBATCH --array=0-1499
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G

datasets=({00..29})
dims=(2 5 10 25 50)
seeds=(0)
ks=(02 03 04 05 06)
exps=(lp_experiment recon_experiment)

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_ks=${#ks[@]}
num_exps=${#exps[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_ks * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_ks * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_ks) % num_seeds ))
k_id=$((SLURM_ARRAY_TASK_ID / num_exps % num_ks ))
exp_id=$((SLURM_ARRAY_TASK_ID % num_exps ))

scale=False
dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
k=${ks[$k_id]}
exp=${exps[$exp_id]}

echo $dataset $dim $seed $k $exp

data_dir=../HEDNet/datasets/synthetic_scale_free/${dataset}
if [ $exp == "recon_experiment" ]
then 
	edgelist=${data_dir}/edgelist.tsv
else 
	edgelist=$(printf ../HEDNet/edgelists/synthetic_scale_free/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
fi

echo edgelist is $edgelist

embedding_dir=embeddings/synthetic_scale_free/${dataset}/${exp}

embedding_dir=$(printf "${embedding_dir}/scale=${scale}/k=${k}/seed=%03d/dim=%03d" ${seed} ${dim})

echo embedding directory is $embedding_dir

if [ ! -f ${embedding_dir}/"mu.csv.gz" ]
then 
	module purge
	module load bluebear

	if [ ! -f ${embedding_dir}/"mu.csv" ]
	then

		module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6

		args=$(echo --edgelist ${edgelist} \
		--embedding ${embedding_dir} --seed ${seed} --dim ${dim} \
		"-k" ${k} "--scale" ${scale})

		python embed.py ${args}
	fi

	echo ${embedding_dir}/"mu.csv" exists compressing
	gzip ${embedding_dir}/"mu.csv"
	gzip ${embedding_dir}/"sigma.csv"

else 

	echo  ${embedding_dir}/"mu.csv.gz" already exists
	echo  ${embedding_dir}/"sigma.csv.gz" already exists

fi