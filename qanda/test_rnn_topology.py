from datasets import *
from fit import *

ds = Datasets(task_index = 1, only_supporting = False, similar_only = False, min_num = 2)
ds.fit()

X, qX, Y = ds.get_training_data()
tX, tXq, tY = ds.get_testing_data()

accuracy_bucket = []
for shs in [100, 500, 1000]:
	for qhs in [50, 100, 200]:
		model_lstm = Fit( vocab_size = ds.vocab_size, sent_hidden_size = shs, query_hidden_size = qhs, epochs = 10 )
		model_lstm.compile_layers()
		model_lstm.run(X, qX, Y)
		model_accu = model_lstm.score(tX, tXq, tY)
		accuracy_bucket.append( (shs, qhs, model_accu) )
print accuracy_bucket