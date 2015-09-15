from datasets import Datasets
from compare_predictions import compare_predictions
from fit import Fit
import cPickle as pkl
import os
os.sys.setrecursionlimit(50000L)


def scan_for_parameters():
	os.sys.setrecursionlimit(50000L)


	# shss = [10, 20, 30, 40, 50, 100]
	shss = [100]
	# qhss = [2,3,10,20,100]
	qhss = [10]
	tasks = [2] #range(2,21)

	for task in tasks:
		for min_num in [1]:#,2,3,4,5,6,7,8]:

			for shs in shss:
				for qhs in qhss:
					ds = Datasets(use_small_target = True
						, task_index = task
						, similar_only = True
						, use10k = False 
						)
					ds.fit()
					X, qX, Y = ds.get_training_data()
					tX, tXq, tY = ds.get_testing_data() 

					model_lstm = Fit( vocab_size = ds.answers_size
						, batch_size =16
						, epochs = 10
						, sent_hidden_size = (shs * task)
						, query_hidden_size = qhs 
						)
					model_lstm.answers = ds.answers
					model_lstm.compile_layers()
					model_lstm.run(X, qX, Y)
					accuracy = model_lstm.score(tX, tXq, tY)

					accum = compare_predictions(ds, model_lstm, tX, tXq, tY)
					model_plus_pred = [accum, model_lstm, accuracy]

					file_name = '../../pickled_models/gensim_hack/model_task{}_using_10k_min_num_{}.pkl'.format(task, min_num)
					print('Pickling model ...')
					with open(file_name,'wb') as f:
						pkl.dump(model_plus_pred, f)

	# dropouts = [0.1, 0.2, 0.4]
	# for dp in dropouts:
	# 	model_lstm = Fit( vocab_size = ds.answers_size
	# 		, batch_size =16
	# 		, epochs = 20
	# 		, dropout = dp
	# 		, sent_hidden_size = 100
	# 		, query_hidden_size = 10 )

	# 	model_lstm.compile_layers()
	# 	model_lstm.run(X, qX, Y)
	# 	accuracy = model_lstm.score(tX, tXq, tY)

	# 	accum = compare_predictions(ds, model_lstm, tX, tXq, tY)
	# 	model_plus_pred = [accum, model_lstm, accuracy]

	# 	file_name = '../../pickled_models/dropouts/model_task1_using_10k_sent_dropout_{}.pkl'.format(dp)
	# 	print('Pickling model ...')
	# 	with open(file_name,'wb') as f:
	# 		pkl.dump(model_plus_pred, f)



if __name__ == "__main__":
	os.sys.setrecursionlimit(50000L)
	scan_for_parameters()