import _pickle as pickle
import  xml.dom.minidom
import math, os, io
import numpy as np

import argparse

parser = argparse.ArgumentParser(
    description='VSM'
)
parser.add_argument('--feedback', default=False, action='store_true')
parser.add_argument('--input_file', default='query-test.xml', metavar='')
parser.add_argument('--output_file', default='pred.csv', metavar='')
parser.add_argument('--model_dir', default='./', metavar='')

args = parser.parse_args()

BASE = 100000
feedback = args.feedback


file = args.input_file
output_file = args.output_file
model = args.model_dir

# - * -  load embedding  - * - # 
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(num) for num in tokens[1:]]
    return data

print('loading embedding..')
emb_dict = load_vectors('/tmp2/Vachel/cc.zh.300.vec')
print('embedding loaded..')


class Vocab:
	def __init__(self):
		self.total = 0
		self.df = {} # {chr_idx: count}
		self.word2i = {} # {chr: chr_idx}
		self.i2word = {} # {chr_idx: chr}
		self.i2doc = {} # {chr_idx: doc_idx}

class Documents:
	def __init__(self):
		self.docs = {} #document class
		self.N = 0 #how many documents.            #^
		self.i2name = {} #doc id to doc name.      #^
		self.L_average = 0 # average doc len

class Document:
	def __init__(self):
		self.len = 0
		self.tf = {}


# - * -  utils  - * - #
def swap(arr, left, right):
	tmp = arr[left]
	arr[left] = arr[right]
	arr[right] = tmp

def quick_sort(poss_docs_id, RSVs,left, right):
	if left < right:
		pivot = RSVs[left]
		l_ptr = left
		r_ptr = right

		while l_ptr < r_ptr:
			while l_ptr < r_ptr and RSVs[r_ptr] < pivot:
				r_ptr -= 1

			while l_ptr < r_ptr and RSVs[l_ptr] >= pivot:
				l_ptr += 1

			if l_ptr == r_ptr:
				swap(RSVs, l_ptr, left)
				swap(poss_docs_id, l_ptr, left)

			else:
				swap(RSVs, l_ptr, r_ptr)
				swap(poss_docs_id, l_ptr, r_ptr)

		quick_sort(poss_docs_id, RSVs, left, l_ptr-1)
		quick_sort(poss_docs_id, RSVs, l_ptr+1, right)

def page_rank(poss_docs_id, RSVs):
	right = len(poss_docs_id) - 1 
	quick_sort(poss_docs_id, RSVs, 0, right)

def possible_docs(querys_id):
	iset_doc_id = set([])
	for q_id in querys_id:
		iset_doc_id = iset_doc_id | set(vocab.i2doc[q_id])

	return list(iset_doc_id)

def cos_similarity(a, b):
	a = np.array(a)
	b = np.array(b)

	pow_a = np.power(a, 2)
	pow_b = np.power(b, 2)

	return (a * b).sum(1) / (np.sqrt(pow_a.sum()) * np.sqrt(pow_b.sum(1)))

def parse_contents(concepts, question = False):
	concepts_id = []

	for concept in concepts:
		data = concept.firstChild.data
		if question:
			data = data[2:]
		l = len(data)
		concept_id = []

		for i in range(l):
			if data[i] in vocab.word2i:
				concept_id.append(vocab.word2i[data[i]])
			if i+1 < l:
				if data[i] in vocab.word2i and data[i+1] in vocab.word2i:
					index = vocab.word2i[data[i]] * BASE + vocab.word2i[data[i+1]]
					if index in vocab.i2doc:
						concept_id.append(index)

		concepts_id.append(concept_id)

	return concepts_id


# - * -  score function  - * - #
#okapi
def RSV(querys_id, doc_id, top_docs_id, tf_q_ids, feedback = False,  dfis = [], VRs = [], VNRs = []):
	rsv = 0
	k_1 = 1.2
	k_3 = 2
	b = 0.75

	for i, q_id in enumerate(querys_id):
		N = documents.N
		df = vocab.df[q_id]

		if q_id not in documents.docs[doc_id].tf:
			continue
		tf = documents.docs[doc_id].tf[q_id]
		L_d = documents.docs[doc_id].len
		L_average = documents.L_average
		q_tf = tf_q_ids[q_id]

		TF = (k_1 + 1) * tf / (k_1 * (( 1 - b) + b*(L_d / L_average)) + tf)
		q_TF = (k_3 + 1) * q_tf / (k_3 + q_tf)
	
		if feedback:
			dfi = dfis[i]
			VR = VRs[i]
			VNR = VNRs[i]
			# print(df, dfi, VR, VNR)
			IDF = math.log((VR + 0.5) / (VNR + 0.5) / ((df - VR + 0.5) / (N - dfi - N//2 + VR + 0.5)))
		else:
			IDF = math.log(N / df)

		rsv += TF * IDF * q_TF
	return rsv

# language model
def possibility(query_ids, doc_id, lamda = 0.9):
	poss = 0
	for i, q_id in enumerate(query_ids):
		corpus_poss = vocab.df[q_id] / vocab.total
		cur_doc = documents.docs[doc_id]
		if q_id not in cur_doc.tf:
			doc_poss = 0
		else:
			doc_poss = cur_doc.tf[q_id] / cur_doc.len

		poss += math.log(lamda * doc_poss + (1 - lamda) * corpus_poss)

	return poss

# language model with word embedding
def possibility_emb(query_ids, doc_id, lamda = 0.6, alpha = 0.3):
	poss = 0
	for i, q_id in enumerate(query_ids):
		corpus_poss = vocab.df[q_id] / vocab.total
		cur_doc = documents.docs[doc_id]
		if q_id not in cur_doc.tf:
			doc_poss = 0
		else:
			doc_poss = cur_doc.tf[q_id] / cur_doc.len

		word_q = vocab.i2word[q_id]
		if word_q not in emb_dict:
			poss += math.log(0.1 * doc_poss + (1 - 0.1) * corpus_poss)
			continue

		emb_q = emb_dict[word_q]

		posses_in_corpus = []
		posses_in_doc = []
		embeddings = []

		for idx, cnt in cur_doc.tf.items():
			word = vocab.i2word[idx]

			if word not in emb_dict:
				continue

			emb = emb_dict[word]
			embeddings.append(emb)

			poss_in_corpus = vocab.df[idx] / vocab.total
			poss_in_doc = cnt / cur_doc.len

			posses_in_corpus.append(poss_in_corpus)
			posses_in_doc.append(poss_in_doc)

		posses_in_corpus = np.array(posses_in_corpus)
		posses_in_doc = np.array(posses_in_doc)

		similaritys = cos_similarity(emb_q, embeddings)
		similarity_sum = similaritys.sum()

		score = (posses_in_corpus * posses_in_doc * similaritys / similarity_sum).sum()
		poss += math.log(lamda * doc_poss + alpha * score + (1 - lamda - alpha) * corpus_poss)

	return poss


# - * -  load class  - * - #
documents = pickle.load(open(os.path.join(model, 'documents.pk'), 'rb'))
vocab = pickle.load(open(os.path.join(model, 'vocab.pk'), 'rb'))



# - * -  parse querys  - * - #
dom = xml.dom.minidom.parse(file)
root = dom.documentElement
titles = root.getElementsByTagName('title')
concepts = root.getElementsByTagName('concepts')
topics = root.getElementsByTagName('number')
questions = root.getElementsByTagName('question')

querys_id = []

topics_id = []
for question in questions:
	data = question.firstChild.data
	query = []
	for i, x in enumerate(data):
		if i >= 2 and x in vocab.word2i:
			query.append(vocab.word2i[x])
	querys_id.append(query)

for topic in topics:
	topic_id = int(topic.firstChild.data[-3:])
	topics_id.append(topic_id)



# - * -  parse contents  - * - #
concepts_id = parse_contents(concepts)
question = parse_contents(questions)

contents_id = []

for concept_id, question_id in zip(concepts_id, question):
	contents_id.append(concept_id + question_id)



# - * -  search  - * - #
retrieved_docs = []
i = 0

for query_id in querys_id:
	tf_q_ids = {}

	for q_id in contents_id[i]:
		if q_id not in tf_q_ids:
			tf_q_ids[q_id] = 1
		else:
			tf_q_ids[q_id] += 1

	poss_docs_id = possible_docs(query_id)
	# RSVs = [RSV(contents_id[i], doc_id, poss_docs_id, tf_q_ids) for doc_id in poss_docs_id]
	# RSVs = [possibility(contents_id[i], doc_id) for doc_id in poss_docs_id]
	RSVs = []

	print('{} docs in total..'.format(len(poss_docs_id)))
	for m, doc_id in enumerate(poss_docs_id):
		RSVs.append(possibility_emb(contents_id[i], doc_id))
		print('{} doc done..'.format(m))
		

	page_rank(poss_docs_id, RSVs)

	if feedback:
		for j in range(1):
			N_ = 1000
			dfis = []
			VRs = []
			VNRs = []

			for q_id in contents_id[i]:
				#doc freq of term i N docs
				dfi = 0
				#doc freq of term i in relative
				VR = 0
				#doc freq of term i in relative
				VNR = 0
				for k, id in enumerate(poss_docs_id[:N_]):
					if q_id in documents.docs[id].tf: 
						if k < N_//2:
							VR += 1
						dfi += 1

				VNR = dfi - VR

				dfis.append(dfi)
				VRs.append(VR)
				VNRs.append(VNR)


			RSVs = [RSV(contents_id[i], doc_id, poss_docs_id, tf_q_ids, True, dfis, VRs, VNRs) for doc_id in poss_docs_id]
			page_rank(poss_docs_id, RSVs)

	print(i)

	i += 1

	retrieved_docs.append([documents.i2name[doc_id] for doc_id in poss_docs_id[:100]])


# output predictions
with open(output_file, 'w') as f:
	k = 0
	f.write('query_id,retrieved_docs\n')
	for i in topics_id:
		docs = ''
		for doc in retrieved_docs[k]:
			docs += doc + ' '

		f.write('{:0>3d},{}\n'.format(i, docs.strip()))
		k += 1








