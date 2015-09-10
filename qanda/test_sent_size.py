from datasets import *
from fit import *
import cPickle as pkl


def test_sent_size( sent = [2,4,6,8, 10] ):
	for i, sent_size in enumerate( sent ):
		ds = Datasets(use_small_target = True, sent_size = sent_size)
		ds.fit()
		X, qX, Y = ds.get_training_data()
		tX, tXq, tY = ds.get_testing_data() 

		shs = [20,30,40,50, 60]

		model_lstm = Fit( vocab_size = ds.answers_size
			, batch_size =64
			, epochs = 50
			, sent_hidden_size = shs[i]
			, query_hidden_size = 3 )

		model_lstm.compile_layers()
		model_lstm.run(X, qX, Y)
		print 'Accuracy for {} story length \n'.format( sent_size ), model_lstm.score(tX, tXq, tY)

		file_name = '../../pickled_models/sent_size/model_lstm_sent_size_{}_task1.pkl'.format( sent_size )
		print('Pickling model ...')
		with open(file_name,'wb') as f:
			pkl.dump(model_lstm, f)


if __name__ == "__main__":
	test_sent_size()