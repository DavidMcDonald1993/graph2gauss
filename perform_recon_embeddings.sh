#!/bin/bash

#SBATCH --job-name=GRAPH2GAUSSembeddingsRECON
#SBATCH --output=GRAPH2GAUSSembeddingsRECON_%A_%a.out
#SBATCH --error=GRAPH2GAUSSembeddingsRECON_%A_%a.err
#SBATCH --array=0-749
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G

scales=(False)
datasets=({cora_ml,citeseer,pubmed,wiki_vote,email})
dims=(2 5 10 25 50)
seeds=({0..29})
ks=(03)

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

data_dir=../HEDNet/datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv
# features=${data_dir}/feats.csv
embedding_dir=embeddings/${dataset}/recon_experiment

embedding_dir=$(printf "${embedding_dir}/scale=${scale}/k=${k}/seed=%03d/dim=%03d/" ${seed} ${dim})
# echo ${embedding_dir}

if [ ! -f ${embedding_dir}"mu.csv.gz" ]
then 
	module purge
	module load bluebear
	module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6

	args=$(echo --edgelist ${edgelist} \
	--embedding ${embedding_dir} --seed ${seed} --dim ${dim} \
	"-k" ${k} "--scale" ${scale})

	# echo $args

	python embed.py ${args}
fi