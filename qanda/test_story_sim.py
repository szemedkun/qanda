from gensim.models.word2vec import Word2Vec
from datasets import *

ds = Datasets(similar_only = True, min_num = 1)
ds.fit()
X, Xq, Y = ds.get_training_data()
tX, tXq, tY = ds.get_testing_data()

#ds._update_word_vec_dict()

# with open('./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt') as f:
# 	lines = f.readlines()

# data = ds.parse_stories(lines)

# glove_file = 'data/glove_data/glove.6B.50d.txt'
# wv = Word2Vec()
# glove = wv.load_word2vec_format(glove_file)

# ws1 = [w.lower() for w in data[1][0][1]]

# ws2 = [w.lower() for w in data[1][1]]


# print glove.n_similarity(ws1, ws2)

