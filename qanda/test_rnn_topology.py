from datasets import *
from fit import *

ds = Datasets(task_index = 1, only_supporting = False, similar_only = False, min_num = 2)
ds.fit()

X, qX, Y = ds.get_training_data()
tX, tXq, tY = ds.get_testing_data()

accuracy_bucket = [('shs', 'qhs', 'shs2', 'qhs2', 'Accuracy:')]
for shs2 in [50, 500]:
	for qhs2 in [50, 200]:
		for shs in [100, 500]:
			for qhs in [50, 200]:
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
				accuracy_bucket.append( (shs, qhs,shs2, qhs2, model_accu) )
print accuracy_bucket


""" TWO LAYER RESULTS


"""





""" SINGLE LAYER RESULTS
[(100, 50, 0.45300000000000001)
, (100, 100, 0.44800000000000001)
, (100, 200, 0.47899999999999998)
, (500, 50, 0.501)
, (500, 100, 0.498)
, (500, 200, 0.52600000000000002)
, (1000, 50, 0.50700000000000001)
, (1000, 100, 0.52600000000000002)
, (1000, 200, 0.52800000000000002)]
"""