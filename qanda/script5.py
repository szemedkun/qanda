from datasets import Datasets
from fit import Fit
from compare_predictions import compare_predictions
import pickle as pkl
import sys

sys.setrecursionlimit(50000)

ds = Datasets(threshold = 0.99
	, task_index = 5
	, only_supporting = False
	, use10k = True
	, use_small_target = True
	, similar_only = True)

ds.fit()

X, qX, Y = ds.get_training_data()

tX, tXq, tY = ds.get_testing_data()

model = Fit(vocab_size = ds.answers_size
	, sent_hidden_size = max( [ min([2 * X.shape[1] // 3, 50]), 2*X.shape[1] // 3 ] )
	, epochs = 20
	, batch_size = 32 )


model.compile_layers()

model.answers = ds.answers

model.run(X, qX, Y)

model.score(tX, tXq, tY)

task5_pred = compare_predictions(ds, model, tX, tXq, tY)

with open('../../capstone_web/static/pickles/task5_pred.pkl', 'wb') as f:
	pkl.dump(task5_pred, f)

