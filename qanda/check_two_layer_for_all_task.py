from datasets import *
from fit import *

accuracy_bag = [('Task Number', 'Accuracy')]
for task in xrange(1,21):
	ds = Datasets(task_index = task, only_supporting = False, similar_only = False, min_num = 2)
	ds.fit()

	X, qX, Y = ds.get_training_data()
	tX, tXq, tY = ds.get_testing_data()

	shs = 500
	shs2 = 500
	qhs = 100
	qhs2 = 100

	model_lstm = Fit( vocab_size = ds.vocab_size
	, sent_hidden_size2 = shs2
	, query_hidden_size2 = qhs2
	, sent_hidden_size = shs
	, query_hidden_size = qhs
	, two_hidden_layers = True
	, epochs = 20 )
	model_lstm.compile_layers()
	model_lstm.run(X, qX, Y)
	model_accu = model_lstm.score(tX, tXq, tY)
	accuracy_bag.append( (task, model_accu) )

print accuracy_bag

""" RESULT


"""