import nltk
from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
import _pickle as pickle
import numpy as np

import csv
import re
import sys
from collections import Counter
csv.field_size_limit(sys.maxsize)
np.random.seed(1)

class Dataset:
	def __init__(self, data_path = 'doc.csv', stop_path = 'stop.txt', tpc_num = 10, min_count = 200, train = True):
		self.data_path = data_path
		self.stop_path = stop_path
		self.tpc_num = tpc_num
		self.min_count = min_count

		self.M_docs_tpc = []
		self.M_tpcs_w = []
		self.bkground = {}
		self.docs = []
		self.docs_num = 0
		self.inverted_file = {}
		
		self.prepare_data(train = train)
		

	def prepare_data(self, train):
		if train:
			docs_data = self.read_data()
			self.build_vocab(docs_data)
			self.initialize()
			self.store()
		else:
			M_docs_tpc = pickle.load(open('M_docs_tpc.pk', 'rb'))
			M_tpcs_w = pickle.load(open('M_tpcs_w.pk', 'rb'))
			bkground = pickle.load(open('bkground.pk', 'rb'))
			docs = pickle.load(open('docs.pk', 'rb'))
			inverted_file = pickle.load(open('inverted_file.pk', 'rb'))
			vocab = pickle.load(open('vocab.pk', 'rb'))
			docs_num = len(docs)

			self.M_docs_tpc = M_docs_tpc
			self.M_tpcs_w = M_tpcs_w
			self.bkground = bkground
			self.docs = docs
			self.docs_num = docs_num
			self.inverted_file = inverted_file
			self.vocab = vocab

	def store(self):
		pickle.dump(self.M_docs_tpc, open('M_docs_tpc.pk', 'wb'))
		pickle.dump(self.M_tpcs_w, open('M_tpcs_w.pk', 'wb'))
		pickle.dump(self.bkground, open('bkground.pk', 'wb'))
		pickle.dump(self.docs, open('docs.pk', 'wb'))
		pickle.dump(self.inverted_file, open('inverted_file.pk', 'wb'))
		pickle.dump(self.vocab, open('vocab.pk', 'wb'))

	def read_data(self):
		docs_data = []
		tokenizer = RegexpTokenizer(r'\w+')

		with open(self.data_path, 'r', encoding='utf8') as f:
			f_csv = csv.reader(f)
			next(f_csv)
			# remove punctuation
			for row in f_csv:
				_, doc = row
				doc = re.sub(r'\\n', ' ', doc.lower())
				tokens = tokenizer.tokenize(doc)
				docs_data.append(tokens)

		self.docs_num = len(docs_data)
		return docs_data

	def build_vocab(self, docs_data):
		# stop words
		stop_words = []
		with open(self.stop_path, 'r') as f:
			for word in f.readlines():
				stop_words.append(word.strip())
		# stop_words = set(stop_words) | set(stopwords.words('english')) 
		stop_words = set(stop_words)

		vocab_cnt = Counter()
		for doc in docs_data:
			for word in doc:
				if word.isdigit():
					vocab_cnt['<num>'] += 1
				else:
					vocab_cnt[word] += 1
		
		# dropout doc by the threshold of min_count and stop words
		total = 0
		vocab = {}
		for key, cnt in vocab_cnt.items():
			if cnt >= self.min_count and key not in stop_words and len(key) > 1:
				vocab[key] = cnt
				total += cnt

		# filter doc and build inverted_file
		docs = []
		for i in range(self.docs_num):
			doc = Counter()
			for word in docs_data[i]:
				if word.isdigit():
					word = '<num>'
				if word in vocab:
					doc[word] += 1
			docs.append(doc)

		inverted_file = {}
		for i, doc in enumerate(docs):
			for word, cnt in doc.items():
				if word.isdigit():
					word = '<num>'
				if word not in inverted_file:
					inverted_file[word] = [i]
				else:
					inverted_file[word].append(i)

		#get P( w_i| B)
		bkground = {}
		for word in vocab:
			bkground[word] = vocab[word] / total


		self.docs = docs
		self.vocab = vocab
		self.vocab_size = len(self.vocab)
		self.bkground = bkground
		self.inverted_file = inverted_file

	def initialize(self):
		utpc_poss = 1/self.tpc_num

		#initialize P( w_i | theta_j)
		for j in range(self.tpc_num):
			tpcj_w = {}
			words_poss = np.random.random(self.vocab_size)
			words_poss = words_poss / np.sum(words_poss)
			for i, word in enumerate(self.vocab.keys()):
				tpcj_w[word] = words_poss[i]
			self.M_tpcs_w.append(tpcj_w)

		#initialize P( theta_j | d_k)
		for k in range(self.docs_num):
			dock_tpcs = []
			for j in range(self.tpc_num):
				dock_tpcs.append(utpc_poss)

			self.M_docs_tpc.append(dock_tpcs)


























