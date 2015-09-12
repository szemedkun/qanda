from datasets import Datasets
from compare_predictions import compare_predictions
from fit import Fit
import cPickle as pkl
import os
os.sys.setrecursionlimit(50000L)


def scan_for_parameters():
	os.sys.setrecursionlimit(50000L)

	ds = Datasets(use_small_target = True, use10k = True)
	ds.fit()
	X, qX, Y = ds.get_training_data()
	tX, tXq, tY = ds.get_testing_data() 

	shss = [10, 20, 30, 40, 50, 100]
	qhss = [2,3,10,20,100]

	for shs in shss:
		for qhs in qhss:
			model_lstm = Fit( vocab_size = ds.answers_size
				, batch_size =16
				, epochs = 20
				, sent_hidden_size = shs
				, query_hidden_size = qhs )

			model_lstm.compile_layers()
			model_lstm.run(X, qX, Y)
			accuracy = model_lstm.score(tX, tXq, tY)

			accum = compare_predictions(ds, model_lstm, tX, tXq, tY)
			model_plus_pred = [accum, model_lstm, accuracy]

			file_name = '../../pickled_models/sent_size/model_lstm_shs_{}_task1_using_10k_qhs_{}.pkl'.format(shs, qhs)
			print('Pickling model ...')
			with open(file_name,'wb') as f:
				pkl.dump(model_plus_pred, f)


if __name__ == "__main__":
	os.sys.setrecursionlimit(50000L)
	scan_for_parameters()