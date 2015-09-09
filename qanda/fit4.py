from datasets import Datasets
from split_x import *

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.layers.core import Dense, Merge
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

class Fit4(object):
	def __init__(self, model = recurrent.LSTM, w2v_dim = 50, sent_hidden_size = 500, 
				query_hidden_size = 100, batch_size = 16, epochs = 10, vocab_size = None, rs = False
				, sent_hidden_size2 = 200, query_hidden_size2 = 50, two_hidden_layers = False):
		'''

		'''
		self.model = model
		self.W2V_DIM = w2v_dim
		self.SENT_HIDDEN_SIZE = sent_hidden_size
		self.QUERY_HIDDEN_SIZE = query_hidden_size
		self.BATCH_SIZE = batch_size
		self.EPOCHS = epochs
		self.vocab_size = vocab_size
		self.SENT_HIDDEN_SIZE2 = sent_hidden_size2
		self.QUERY_HIDDEN_SIZE2 = query_hidden_size2
		self.two_hidden_layers = two_hidden_layers
		self.rs = rs # return sequence

	def compile_layers(self):
		'''
		Ones I am done with exploration, I will make this flexible!
		'''
		print('Build model...')
		RNN = self.model

		# statement1
		sentrnn1 = Sequential()
		sentrnn1.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement2
		sentrnn2 = Sequential()
		sentrnn2.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement3
		sentrnn3 = Sequential()
		sentrnn3.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement4
		sentrnn4 = Sequential()
		sentrnn4.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement5
		sentrnn5 = Sequential()
		sentrnn5.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement6
		sentrnn6 = Sequential()
		sentrnn6.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement7
		sentrnn7 = Sequential()
		sentrnn7.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement8
		sentrnn8 = Sequential()
		sentrnn8.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement9
		sentrnn9 = Sequential()
		sentrnn9.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# statement10
		sentrnn10 = Sequential()
		sentrnn10.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))

		# query
		qrnn = Sequential()
		qrnn.add(RNN(self.W2V_DIM, self.QUERY_HIDDEN_SIZE, return_sequences=self.rs))
		#qrnn.add(RNN(self.QUERY_HIDDEN_SIZE, self.QUERY_HIDDEN_SIZE2, return_sequences = False))
		
		# merging just stories
		model_story = Sequential()
		model_story.add(Merge([sentrnn1
						, sentrnn2
						, sentrnn3
						, sentrnn4
						, sentrnn5
						, sentrnn6
						, sentrnn7
						, sentrnn8
						, sentrnn9
						, sentrnn10], mode='concat'))

		# A layer before merging with query
		model_story.add(Dense(10*self.SENT_HIDDEN_SIZE, 50, activation = 'relu'))

		# Second merge with query
		model = Sequential()
		model.add(Merge([model_story, qrnn], mode='concat'))

		# output layer
		model.add(Dense(50+self.QUERY_HIDDEN_SIZE, self.vocab_size, activation='softmax'))

		model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
		self.model = model


	def run(self, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Xq, Y):
		'''
		Ones I am done with exploration, I will make this more flexible

		'''
		print('Training')
		self.model.fit([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Xq], Y, batch_size=self.BATCH_SIZE, nb_epoch=self.EPOCHS, validation_split=0.05, show_accuracy=True)

	def score(self, tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8, tX9, tX10, tXq, tY):
		'''
		Come back to this and make it flexible

		Right now it returns accuracy
		'''
		loss, acc = self.model.evaluate([tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8, tX9, tX10, tXq], tY, batch_size=self.BATCH_SIZE, show_accuracy=True)
		print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
		return acc


if __name__ == "__main__":
	ds = Datasets(task_index = 1, only_supporting = False, use10k = False)
	ds.fit()

	X, qX, Y = ds.get_training_data()
	tX, tXq, tY = ds.get_testing_data()

	train = ds.train
	test = ds.test

	max_doc_len = ds.story_maxlen
	ind_list = get_indices( train, max_doc_len )
	all_parts = split_x( X, ind_list )
	new_dict = make_x( all_parts )
	X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = new_dict[0], \
											  new_dict[1], \
											  new_dict[2], \
											  new_dict[3], \
											  new_dict[4], \
											  new_dict[5], \
											  new_dict[6], \
											  new_dict[7], \
											  new_dict[8], \
											  new_dict[9]
	#max_doc_len = ds.query_maxlen
	ind_list = get_indices( test, max_doc_len )
	all_parts = split_x( tX, ind_list )
	new_dict = make_x( all_parts )
	tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8, tX9, tX10 = new_dict[0], \
												  new_dict[1], \
												  new_dict[2], \
												  new_dict[3], \
												  new_dict[4], \
												  new_dict[5], \
												  new_dict[6], \
												  new_dict[7], \
												  new_dict[8], \
												  new_dict[9]

	model4 = Fit4( vocab_size = ds.vocab_size, sent_hidden_size = 1000, query_hidden_size = 20
	, model = recurrent.LSTM )
	model4.compile_layers()
	model4.run(X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, qX, Y)
	acc = model4.score(tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8, tX9, tX10, tXq, tY)
	y_pred = model4.model.predict_classes([tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8, tX9, tX10, tXq], 
		batch_size = model4.BATCH_SIZE)