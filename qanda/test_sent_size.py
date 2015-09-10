from datasets import *
from split_x import *
from fit import *
from fit2 import *
import cPickle as pkl
import os
os.sys.setrecursionlimit(50000L)


def test_sent_size( sent = [2,4,6,8, 10] ):
	os.sys.setrecursionlimit(50000L)

	for i, sent_size in enumerate( sent ):
		i, sent_size = 0, 2
		ds = Datasets(use_small_target = True, sent_size = sent_size)
		ds.fit()
		X, qX, Y = ds.get_training_data()
		tX, tXq, tY = ds.get_testing_data() 

		shs = [20,30,40,50, 60]

		model_lstm = Fit( vocab_size = ds.answers_size
			, batch_size =16
			, epochs = 20
			, sent_hidden_size = 50
			, query_hidden_size = 10 )

		model_lstm.compile_layers()
		model_lstm.run(X, qX, Y)
		print 'Accuracy for {} story length \n'.format( sent_size ), model_lstm.score(tX, tXq, tY)

		file_name = '../../pickled_models/sent_size/model_lstm_sent_size_{}_task1_qz_10.pkl'.format( sent_size )
		print('Pickling model ...')
		with open(file_name,'wb') as f:
			pkl.dump(model_lstm, f)

def test_sent_size2( sent_size = 2):
	'''
	tests if splitting helps
	'''
	sent_size = 2
	ds = Datasets(use_small_target = True, sent_size = sent_size)
	ds.fit()
	X, qX, Y = ds.get_training_data()
	tX, tXq, tY = ds.get_testing_data() 

	test = ds.test
	train = ds.train

	max_doc_len = ds.story_maxlen
	ind_list = get_indices( train, max_doc_len )
	all_parts = split_x( X, ind_list, num_split = 2  )
	new_dict = make_x( all_parts )
	X1, X2 = new_dict[0], new_dict[1]

	ind_list = get_indices( test, max_doc_len )
	all_parts = split_x( X, ind_list, num_split =2  )
	new_dict = make_x( all_parts )
	tX1, tX2 = new_dict[0], new_dict[1]

	model_lstm = Fit2( vocab_size = ds.answers_size
	, batch_size =16
	, epochs = 20
	, sent_hidden_size = 50
	, query_hidden_size = 10 )

	model_lstm.compile_layers()
	model_lstm.run(X1, X2, qX, Y)
	print 'Accuracy for {} story length \n'.format( sent_size ), model_lstm.score(tX1, tX2, tXq, tY)



if __name__ == "__main__":
	os.sys.setrecursionlimit(50000L)
	test_sent_size2()
	# X, Xq, Y = ds.get_training_data()
	# train = ds.train
	# max_doc_len = ds.story_maxlen
	# ind_list = get_indices( train, max_doc_len )
	# all_parts = split_x( X, ind_list, num_split  )
	# new_dict = make_x( all_parts )