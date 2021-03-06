from datasets import Datasets
import numpy as np
# np.random.seed(1337)  # for reproducibility
from keras.layers.core import Dense, Merge, Dropout
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


class Fit:
    def __init__(self,
                 model=recurrent.LSTM,
                 w2v_dim=50,
                 sent_hidden_size=500,
                 dropout=None,
                 query_hidden_size=100,
                 batch_size=16,
                 epochs=10, vocab_size=None, rs=False,
                 sent_hidden_size2=200, query_hidden_size2=50,
                 two_hidden_layers=False):
        '''

        '''
        self.model = recurrent.LSTM
        self.W2V_DIM = w2v_dim
        self.SENT_HIDDEN_SIZE = sent_hidden_size
        self.QUERY_HIDDEN_SIZE = query_hidden_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.vocab_size = vocab_size
        self.SENT_HIDDEN_SIZE2 = sent_hidden_size2
        self.QUERY_HIDDEN_SIZE2 = query_hidden_size2
        self.two_hidden_layers = two_hidden_layers
        self.rs = rs
        self.dropout = dropout
        self.X = None
        self.Xq = None
        self.Y = None
        self.answers = None

    def compile_layers(self):
        '''
        Ones I am done with exploration, I will make this flexible!
        '''
        if self.two_hidden_layers:
            print('Build model...')
            RNN = self.model
            # statements lstm
            sentrnn = Sequential()
            sentrnn.add(RNN(self.W2V_DIM,
                        self.SENT_HIDDEN_SIZE,
                        return_sequences=True))
            sentrnn.add(Dense(self.SENT_HIDDEN_SIZE,
                        self.SENT_HIDDEN_SIZE2,
                        activation='relu'))

            # query lstm
            qrnn = Sequential()
            qrnn.add(RNN(self.W2V_DIM, self.QUERY_HIDDEN_SIZE, return_sequences=True))
            qrnn.add(RNN(self.QUERY_HIDDEN_SIZE, self.QUERY_HIDDEN_SIZE2, return_sequences = False))

            # merging
            model = Sequential()
            model.add(Merge([sentrnn, qrnn], mode='concat'))

            # output layer
            model.add(Dense(self.SENT_HIDDEN_SIZE2 + self.QUERY_HIDDEN_SIZE2, self.vocab_size, activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
            self.model = model
        else:
            print('Build model...')
            RNN = self.model
            # statements lstm
            sentrnn = Sequential()
            sentrnn.add(RNN(self.W2V_DIM, self.SENT_HIDDEN_SIZE, return_sequences=self.rs))
            if self.dropout:
                sentrnn.add(Dropout(self.dropout))
            # sentrnn.add(RNN(self.SENT_HIDDEN_SIZE, self.SENT_HIDDEN_SIZE2, return_sequences = False))

            # query lstm
            qrnn = Sequential()
            qrnn.add(RNN(self.W2V_DIM, self.QUERY_HIDDEN_SIZE, return_sequences=self.rs))
            #qrnn.add(RNN(self.QUERY_HIDDEN_SIZE, self.QUERY_HIDDEN_SIZE2, return_sequences = False))

            # merging
            model = Sequential()
            model.add(Merge([sentrnn, qrnn], mode='concat'))

            # model.add

            # output layer
            model.add(Dense(self.SENT_HIDDEN_SIZE + self.QUERY_HIDDEN_SIZE, self.vocab_size, activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
            self.model = model


    def run(self, X, Xq, Y):
        '''
        Ones I am done with exploration, I will make this more flexible

        '''
        print('Training')
        self.X = X
        self.Xq = Xq
        self.Y = Y
        self.model.fit([X, Xq], Y, batch_size=self.BATCH_SIZE, nb_epoch=self.EPOCHS, show_accuracy=True, validation_split = 0.1)

    def score(self,tX, tXq, tY):
        '''
        Come back to this and make it flexible

        Right now it returns accuracy
        '''
        loss, acc = self.model.evaluate([tX, tXq], tY, batch_size=self.BATCH_SIZE, show_accuracy=True)
        print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
        return acc

if __name__ == "__main__":
    # ds = Datasets(task_index = 1, only_supporting = False, use10k = False, use_small_target = True)
    # ds.fit()

    # X, qX, Y = ds.get_training_data()
    # tX, tXq, tY = ds.get_testing_data()

    # model_lstm = Fit( vocab_size = ds.answers_size, batch_size =16, epochs = 10, sent_hidden_size = 100, query_hidden_size = 10 )
    # model_lstm.compile_layers()
    # model_lstm.run(X, qX, Y)
    # print model_lstm.score(tX, tXq, tY)
    pass
