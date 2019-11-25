#!/bin/bash

for dataset in {00..29}
do
	for dim in 2 5 10 25 50
	do	
		for seed in 0
		do
			for exp in recon_experiment lp_experiment
			do

                for scale in True False
                do 

                    for k in 02 03 04 05 06
                    do 

                        embedding_dir=$(printf embeddings/synthetic_scale_free/${dataset}/${exp}/scale=${scale}/k=${k}/seed=%03d/dim=%03d ${seed} ${dim})

                        if [ -f ${embedding_dir}/mu.csv ] 
                        then
                            if [ ! -f ${embedding_dir}/mu.csv.gz ]
                            then 
                                gzip ${embedding_dir}/mu.csv
                            fi
                        else
                            echo no embedding at ${embedding_dir}/mu.csv
                        fi

                        if [ -f ${embedding_dir}/sigma.csv ]
                        then 
                            if [ ! -f ${embedding_dir}/sigma.csv.gz ]
                            then 
                                gzip ${embedding_dir}/sigma.csv
                            fi
                        else
                            echo no variance at ${embedding_dir}/sigma.csv
                        fi


                    done 


                done 
            done
        done 
    done 
done 
    