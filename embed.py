import numpy as np
import networkx as nx 
import pandas as pd
import scipy.sparse as sp



from g2g.model import Graph2Gauss

import argparse
import os

import pickle as pkl


from g2g.utils import sparse_feeder
import tensorflow as tf

def load_data(args):

	edgelist_filename = args.edgelist
	features_filename = args.features
	labels_filename = args.labels

	graph = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=int,
		create_using=nx.DiGraph() if args.directed else nx.Graph())

	zero_weight_edges = [(u, v) for u, v, w in graph.edges(data="weight") if w == 0.]
	print ("removing", len(zero_weight_edges), "edges with 0. weight")
	graph.remove_edges_from(zero_weight_edges)

	print ("ensuring all weights are positive")
	nx.set_edge_attributes(graph, name="weight", values={edge: abs(weight) 
		for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

	print ("number of nodes: {}\nnumber of edges: {}\n".format(len(graph), len(graph.edges())))

	if features_filename is not None:

		print ("loading features from {}".format(features_filename))

		if features_filename.endswith(".csv"):
			features = pd.read_csv(features_filename, index_col=0, sep=",")
			features = [features.reindex(sorted(graph)).values, 
				features.reindex(sorted(features.index)).values]
			print ("no scaling applied")
			# scaler = StandardScaler()
			# scaler.fit(features[0])
			# features = list(map(scaler.transform,
			# 	features))
		else:
			raise Exception

		from scipy.sparse import csr_matrix
		features = tuple(map(csr_matrix, features))

		print ("training features shape is {}".format(features[0].shape))
		print ("all features shape is {}\n".format(features[1].shape))

	else: 
		features = None

	if labels_filename is not None:

		print ("loading labels from {}".format(labels_filename))

		if labels_filename.endswith(".csv"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(sorted(graph.nodes())).values.astype(int)#.flatten()
			assert len(labels.shape) == 2
		elif labels_filename.endswith(".pkl"):
			with open(labels_filename, "rb") as f:
				labels = pkl.load(f)
			labels = np.array([labels[n] for n in sorted(graph.nodes())], dtype=np.int)
		else:
			raise Exception

		print ("labels shape is {}\n".format(labels.shape))

	else:
		labels = None

	graph = nx.convert_node_labels_to_integers(graph, 
		ordering="sorted")

	return graph, features, labels

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="HEAT algorithm for feature learning on complex networks")

	parser.add_argument("--edgelist", dest="edgelist", type=str, default=None,
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, default=None,
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, default=None,
		help="path to labels")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 10).", default=10)

	parser.add_argument("-k", dest="k", type=int,
		help="Maximum context size to consider (K in original paper) (default is 3)", 
		default=3)

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")
	
	parser.add_argument("--embedding", dest="embedding_path", default=None, 
		help="path to save embedings.")

	parser.add_argument('--scale', dest="scale", type=str, default="False",
		help='flag to scale pairs')

	args = parser.parse_args()
	return args

def main():

	args = parse_args()
	args.directed = True

	if not os.path.exists(args.embedding_path):
		print ("making", args.embedding_path)
		os.makedirs(args.embedding_path, exist_ok=True)

	graph, features, _ = load_data(args)

	A = nx.adjacency_matrix(graph, nodelist=sorted(graph))
	
	if features is None:
		print ("using identity features")
		features = [sp.csr_matrix(sp.identity(len(graph)))] * 2
	else: 
		assert isinstance(features, tuple)
		assert len(features) == 2
	X = features[0]
	if not isinstance(X, sp.csr_matrix):
		X = sp.csr_matrix(X)

	g2g = Graph2Gauss(A=A, X=X, L=args.embedding_dim, 
		K=args.k, verbose=True, p_val=0.0, p_test=0.0, p_nodes=0,
		seed=args.seed, scale=args.scale=="True")
	sess = g2g.train()

	# predict using entire features matrix
	all_feats = sparse_feeder(features[1])

	mu, sigma = sess.run([g2g.mu, g2g.sigma],
		feed_dict={g2g.X: all_feats})

	mu_filename = os.path.join(args.embedding_path, 
		"mu.csv")
	sigma_filename = os.path.join(args.embedding_path, 
		"sigma.csv")

	mu_df = pd.DataFrame(mu, )
	sigma_df = pd.DataFrame(sigma, )

	print ("saving mu to", mu_filename)
	mu_df.to_csv(mu_filename)

	print ("saving sigma to", sigma_filename)
	sigma_df.to_csv(sigma_filename)

if __name__ == "__main__":
	main()