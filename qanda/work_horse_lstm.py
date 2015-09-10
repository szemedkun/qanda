import cPickle as pkl
from datasets import *
from fit3 import *
from split_x import *

def work_horse_lstm():
	for task in xrange(1,21):
		ds = Datasets(task_index = task, only_supporting = False, use10k = False)
		ds.fit()

		X, qX, Y = ds.get_training_data()
		tX, tXq, tY = ds.get_testing_data()

		train = ds.train
		test = ds.test

		max_doc_len = ds.story_maxlen
		ind_list = get_indices( train, max_doc_len )
		all_parts = split_x( X, ind_list )
		new_dict = make_x( all_parts )
		X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = new_dict[0], \
												  new_dict[1], \
												  new_dict[2], \
												  new_dict[3], \
												  new_dict[4], \
												  new_dict[5], \
												  new_dict[6], \
												  new_dict[7], \
												  new_dict[8], \
												  new_dict[9]
		#max_doc_len = ds.query_maxlen
		ind_list = get_indices( test, max_doc_len )
		all_parts = split_x( tX, ind_list )
		new_dict = make_x( all_parts )
		tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8, tX9, tX10 = new_dict[0], \
													  new_dict[1], \
													  new_dict[2], \
													  new_dict[3], \
													  new_dict[4], \
													  new_dict[5], \
													  new_dict[6], \
													  new_dict[7], \
													  new_dict[8], \
													  new_dict[9]

		model_lstm = Fit3( vocab_size = ds.vocab_size, batch_size =32, epochs=20, model = recurrent.LSTM )
		model_lstm.compile_layers()
		model_lstm.run(X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, qX, Y)
		acc = model_lstm.score(tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8, tX9, tX10, tXq, tY)

		file_name = '../../pickled_models/model_lstm'+str(task)+'_merged.pkl'
		with open(file_name, 'wb') as f:
			pkl.dump(model_lstm, f)

if __name__ == "__main__":
	work_horse_lstm()