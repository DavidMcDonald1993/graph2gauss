import pandas as pd 
import os
import itertools

from pandas.errors import EmptyDataError

def main():

    datasets = ("twitter", "gplus")
    feats = ("nofeats", "feats")
    dims = ( 5, 10, 25, 50)
    seeds = range(30)
    exps = ["recon_experiment", "lp_experiment", "rn_experiment"]
    ks = (1, )
    matrices = ["mu", "sigma"]
    scales = ("False", )

    for dataset, feat, dim, seed, exp, k, matrix, scale in itertools.product(
        datasets, feats, dims, seeds, exps, ks, matrices, scales
    ):
        embedding_directory = os.path.join(
            "embeddings", 
            dataset, feat, exp, 
            "scale={}".format(scale),
            "k={:02d}".format(k),
            "seed={:03d}".format(seed),
            "dim={:03d}".format(dim)
        )

        filename = os.path.join(embedding_directory, 
            "{}.csv.gz".format(matrix))

        try:
            pd.read_csv(filename)
        except EmptyDataError:
            print (filename, "is empty removing it")
            os.remove(filename)
        except IOError:
            print (filename, "does not exist")




if __name__ == "__main__":
    main()