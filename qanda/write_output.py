import numpy as np
import csv
from datasets import *
from fit import *

task = 1
ds = Datasets(task_index = task, only_supporting = False, similar_only = False, min_num = 2)
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

correct_file = '../output/tast_' + str( task ) + '_correct.csv'
incorrect_file = '../output/tast_' + str( task ) + '_incorrect.csv'

correct_output = [['sent+quest', 'actual_ans', 'pred_ans']]
incorrect_output = [['sent+quest', 'actual_ans', 'pred_ans']]
for ind in xrange(1000):
	line = ' '.join(ds.test[ind][0]) + ' '.join(ds.test[ind][1])
	true_ans = ds.test[ind][2]
	pred_ans = vocab[y_pred[ind] - 1]
	co = csv.writer( open( correct_file, 'wb' ) )
	io = csv.writer( open( incorrect_file, 'wb' ) )
	if true_ans == pred_ans:
		correct_output.append( [line, true_ans, pred_ans] )
		co.writerow( correct_output )
	else:
		incorrect_output.append( [line, true_ans, pred_ans] )
		io.writerow( incorrect_output )
