from EM_2 import EM
import _pickle as pickle
from preprocessing import Dataset
import numpy as np

import argparse

parser = argparse.ArgumentParser(
    description='pLSA'
)
parser.add_argument('--seed', default=False, action='store_true')
parser.add_argument('--input_file', metavar='')
parser.add_argument('--output_file', metavar='')
parser.add_argument('--group_file', metavar='')

args = parser.parse_args()

input_path = args.input_file
output_path = args.output_file
group_path = args.group_file

seed_path = 'seeds.npy'
model_path = 'em_0.62.pk'

# sum(P(w_i | theta_j) * P (theta_j | d_k), j) / len_class[]
def score(doc_id, cls):
	poss = 0
	seed = seeds[cls]
	length = len_class[cls]

	for word in seed:
		for j in range(tpc_num):
			poss += em.M_docs_tpc[doc_id][j] * em.M_tpcs_w[j][word]

	poss /= length

	return poss

def classify(doc_id):
	return np.argmax([score(doc_id, i) for i in range(class_num)])

def read_seeds(group_path):
	seeds = []
	len_class = []

	with open(group_path, 'r') as f:
		f.readline()
		for row in f.readlines():
			_, name, word = row.split(',')
			seed = name.split('.')
			seed.append(word.strip())

			seed_filted = [word for word in seed if word in dataset.bkground]

			seeds.append(seed_filted)
			len_class.append(len(seed_filted))

	if args.seed:	
		seeds = np.load(seed_path)
		seeds_filted = []
		for row in seeds:
		 	seed_filted = [word for word in row if word in dataset.bkground]
		 	seeds_filted.append(seed_filted)
		len_class = [len(seed) for seed in seeds_filted]

		return seeds_filted, len_class

	return seeds, len_class


dataset = Dataset(data_path = input_path, train = True)
docs = dataset.docs
docs_num = dataset.docs_num

em = pickle.load(open(model_path, 'rb'))
tpc_num = len(em.M_tpcs_w)

# read seeds
class_num = 17
seeds, len_class = read_seeds(group_path)

res = []
for i in range(docs_num):
	res.append(classify(i))

# output
with open(output_path, 'w') as f:
	f.write('doc_id,class_id\n')
	for i in range(docs_num):
		f.write('{},{}\n'.format(i, res[i]))















