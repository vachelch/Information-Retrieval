import _pickle as pickle
import io, os

import argparse
parser = argparse.ArgumentParser(
    description='VSM'
)
parser.add_argument('--model_dir', default='./', metavar='')
args = parser.parse_args()

model = args.model_dir

BASE = 100000

class Vocab:
	def __init__(self):
		self.total = 0
		self.df = {} # {chr_idx: count}
		self.word2i = {} # {chr: chr_idx}
		self.i2word = {} # {chr_idx: chr}
		self.i2doc = {} # {chr_idx: doc_idx}

class Documents:
	def __init__(self):
		self.docs = {} # {doc_idx: class Document}
		self.N = 0 # how many documents.            #^
		self.i2name = {} # doc_idx to doc_name.      #^
		self.L_average = 0 # average doc len

class Document:
	def __init__(self):
		self.len = 0 # doc len
		self.tf = {} # {chr_idx: count} in a doc


vocab = Vocab()
documents = Documents()

print('read file: \'vocab.all\'')
with io.open(os.path.join(model, 'vocab.all'), mode="r", encoding="utf-8") as f:
	i = 0
	for row in f.readlines():
		vocab.word2i[row.strip()] = i
		vocab.i2word[i] = row.strip()
		i += 1

print('read file: \'file-list\'')
with open(os.path.join(model, 'file-list'), 'r') as f:
	i = 0
	for row in f.readlines():
		row = row.strip()
		doc_name = row.split('/')[-1].lower()
		documents.i2name[i] = doc_name
		i += 1
	documents.N = i

print('read file: \'inverted-file\'')
with open(os.path.join(model, 'inverted-file'), 'r') as f:
	line = f.readline()
	while line:
		id1, id2, n = [int(x) for x in line.split()]
		vocab.total += n
		if id2 == -1:
			vocab.df[id1] = n
			vocab.i2doc[id1] = []
			for i in range(n):
				row = f.readline()
				doc_id, count = [int(x) for x in row.split()]

				vocab.i2doc[id1].append(doc_id)

				if doc_id not in documents.docs:
					doc = Document()
					doc.tf[id1] = count
					doc.len += count

					documents.docs[doc_id] = doc
				else:
					documents.docs[doc_id].tf[id1] = count
					documents.docs[doc_id].len += count

		else:
			vocab.df[id1*BASE + id2] = n
			vocab.i2doc[id1*BASE + id2] = []

			word = vocab.i2word[id1] + vocab.i2word[id2]
			vocab.word2i[word] = id1*BASE + id2
			vocab.i2word[id1*BASE + id2] = word

			for i in range(n):
				row = f.readline()
				doc_id, count = [int(x) for x in row.split()]

				vocab.i2doc[id1*BASE + id2].append(doc_id)

				if doc_id not in documents.docs:
					doc = Document()
					doc.tf[id1*BASE + id2] = count

					documents.docs[doc_id] = doc
				else:
					documents.docs[doc_id].tf[id1*BASE + id2] = count

		line = f.readline()

sm = 0
for doc_id, doc in documents.docs.items():
	sm += doc.len
documents.L_average  = sm / documents.N


pickle.dump(vocab, open(os.path.join(model, 'vocab.pk'), 'wb'))
pickle.dump(documents, open(os.path.join(model, 'documents.pk'), 'wb'))







