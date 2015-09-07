import numpy as np
import csv
from datasets import *
from fit import *

task = 2
ds = Datasets(task_index = task, only_supporting = False, 
	similar_only = True, use10k = False, use_tree = False, min_num = 2)
ds.fit()

X, qX, Y = ds.get_training_data()
tX, tXq, tY = ds.get_testing_data()


shs = 500
#shs2 = 500
qhs = 100
#qhs2 = 100

model_lstm = Fit( vocab_size = ds.vocab_size
#, sent_hidden_size2 = shs2
#, query_hidden_size2 = qhs2
, sent_hidden_size = shs
, query_hidden_size = qhs
, two_hidden_layers = False
, epochs = 10 )
model_lstm.compile_layers()
model_lstm.run(X, qX, Y)
model_accu = model_lstm.score(tX, tXq, tY)
vocab = ds.vocab
y_pred = model_lstm.model.predict_classes( [tX, tXq], batch_size = model_lstm.BATCH_SIZE )
