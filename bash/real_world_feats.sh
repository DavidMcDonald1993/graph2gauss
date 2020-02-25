#!/bin/bash

#SBATCH --job-name=G2GFEATS
#SBATCH --output=G2GFEATS_%A_%a.out
#SBATCH --error=G2GFEATS_%A_%a.err
#SBATCH --array=0-1349
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=25G

datasets=(cora_ml citeseer pubmed)
dims=(2 5 10 25 50)
seeds=({0..29})
ks=(03)
exps=(lp_experiment recon_experiment rn_experiment)

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

data_dir=../HEADNET/datasets/${dataset}
if [ $exp == "recon_experiment" ]
then 
	edgelist=${data_dir}/edgelist.tsv.gz
elif [ $exp == "rn_experiment" ]
then 
    edgelist=$(printf ../HEADNET/nodes/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
else 
	edgelist=$(printf ../HEADNET/edgelists/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
fi

echo edgelist is $edgelist

features=${data_dir}/feats.csv.gz

embedding_dir=embeddings/${dataset}/feats/${exp}

embedding_dir=$(printf "${embedding_dir}/scale=${scale}/k=${k}/seed=%03d/dim=%03d" ${seed} ${dim})

echo embedding directory is $embedding_dir


if [ ! -f ${embedding_dir}/"mu.csv.gz" ]
then 
	module purge
	module load bluebear

	if [ ! -f ${embedding_dir}/"mu.csv" ]
	then

		module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6

		args=$(echo --edgelist ${edgelist} --features ${features} \
		--embedding ${embedding_dir} --seed ${seed} --dim ${dim} \
		"-k" ${k} "--scale" ${scale})

		python embed.py ${args}

	fi

	echo ${embedding_dir}/"mu.csv" exists compressing
	gzip ${embedding_dir}/"mu.csv"
	gzip ${embedding_dir}/"sigma.csv"

else

	echo ${embedding_dir}/"mu.csv.gz" already exists
	echo ${embedding_dir}/"sigma.csv.gz" already exists

fi