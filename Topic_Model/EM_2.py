from preprocessing import Dataset
import _pickle as pickle

class EM:
	def __init__(self, M_docs_tpc, M_tpcs_w):
		self.M_docs_tpc = M_docs_tpc
		self.M_tpcs_w = M_tpcs_w

min_count = 200
tpc_num = 50
data_path = 'doc.csv'
stop_path = 'stop.txt'
epoch = 100

def main():
	dataset = Dataset(data_path, stop_path, tpc_num, min_count, True)
	print('data loaded')

	M_tpcs_w = dataset.M_tpcs_w
	M_docs_tpc = dataset.M_docs_tpc
	bkground = dataset.bkground
	docs = dataset.docs
	docs_num = dataset.docs_num
	inverted_file = dataset.inverted_file

	# initialzie
	E_tpc = []
	for k in range(docs_num):
		doc = {}
		for word in docs[k].keys():
			doc[word] = [0 for k in range(tpc_num)]
		E_tpc.append(doc)


	for i in range(epoch):
		print('E step', i)
		########  E_step  ########
		# Estimate P(theta_j | w_i, d_k)
		for k in range(docs_num): 
			for word, cnt in docs[k].items():
				# P(theta_j | w_i, d_k)
				dock_wi = 0
				for j in range(tpc_num):
					E_tpc[k][word][j] = M_docs_tpc[k][j] * M_tpcs_w[j][word]
					dock_wi += E_tpc[k][word][j]
				
				for j in range(tpc_num):
					E_tpc[k][word][j] /= dock_wi


		print('M step', i)
		########  M_step  ########
		# Maximize P( w_i | theta_j), P( theta_j | d_k) and lambda

		# step1: Maximize P( w_i | theta_j) 
		total_topics = 0

		for j in range(tpc_num):
			total_tpcj = 0
			for word in bkground.keys():
				M_tpcs_w[j][word] = 0
				for doc_id in inverted_file[word]:
					M_tpcs_w[j][word] += docs[doc_id][word] * E_tpc[doc_id][word][j]
				total_tpcj += M_tpcs_w[j][word]

			# normalize
			for word in bkground.keys():
				M_tpcs_w[j][word] /= total_tpcj

			total_topics += total_tpcj


		# step2: Maximize P( theta_j | d_k)
		for k in range(docs_num):
			total_dock = 0
			for j in range(tpc_num):
				M_docs_tpc[k][j] = 0
				for word, cnt in docs[k].items():
					M_docs_tpc[k][j] += cnt * E_tpc[k][word][j]

				total_dock += M_docs_tpc[k][j]

			# normalize
			for j in range(tpc_num):
				M_docs_tpc[k][j] /= total_dock


	em = EM(M_docs_tpc, M_tpcs_w)
	pickle.dump(em, open('em2.pk', 'wb'))

if __name__ == '__main__':
	main()












