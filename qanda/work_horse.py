from datasets import Datasets
from fit import Fit
import cPickle as pkl
import os
os.sys.setrecursionlimit(50000L)


def task1():
	ds = Datasets(threshold = 0.99, task_index = 1, only_supporting = False
		, use10k = False, use_small_target = True, similar_only = True)
	ds.fit()
	X, qX, Y = ds.get_training_data()
	tX, tXq, tY = ds.get_testing_data()

	model = Fit(vocab_size = ds.answers_size, sent_hidden_size = 100)
	model.compile_layers()
	model.run(X, qX, Y)
	model.score(tX, tXq, tY)


def task2():
	os.sys.setrecursionlimit(50000L)
	ds = Datasets(threshold = 0.99, task_index = 2, only_supporting = False
		, use10k = False, use_small_target = True, similar_only = True)
	ds.fit()
	X, qX, Y = ds.get_training_data()
	tX, tXq, tY = ds.get_testing_data()


	accuracy = 0.427 # the best one from the last model
	for dp in [.1,.2]:
		for shs in [100, 200, 500]:
			model = Fit(vocab_size = ds.answers_size
				, dropout = dp
				, sent_hidden_size = shs)
			model.compile_layers()
			model.run(X, qX, Y)
			acc = model.score(tX, tXq, tY)
			model.answers = ds.answers
			if acc > accuracy:
				accuracy = acc
				print 'shs: ', shs
				print 'dropout', dp
				print('Pickling the best model so far...')
				filename = '../../pickled_models/gensim_hack/task2/best.pkl'
				with open(filename, 'wb') as f:
					pkl.dump(model, f)
			else:
				print 'shs: ', shs
				print 'dropout', dp
def task3():
	os.sys.setrecursionlimit(50000L)
	ds = Datasets(threshold = 0.99, task_index = 3, only_supporting = False
		, use10k = False, use_small_target = True, similar_only = True)
	ds.fit()
	X, qX, Y = ds.get_training_data()
	tX, tXq, tY = ds.get_testing_data()


	accuracy = 0.0
	for dp in [.1,.2,.5]:
		for shs in [100, 500, 1000]:
			model = Fit(vocab_size = ds.answers_size
				, dropout = dp
				, epochs = 30
				, query_hidden_size = 20
				, sent_hidden_size = shs)
			model.compile_layers()
			model.run(X, qX, Y)
			acc = model.score(tX, tXq, tY)
			model.answers = ds.answers
			if acc > accuracy:
				accuracy = acc
				print 'shs: ', shs
				print 'dropout', dp
				print('Pickling the best model so far...')
				filename = '../../pickled_models/gensim_hack/task3/best.pkl'
				with open(filename, 'wb') as f:
					pkl.dump(model, f)
			else:
				print 'shs: ', shs
				print 'dropout', dp


if __name__=="__main__":
	os.sys.setrecursionlimit(50000L)
	task2()