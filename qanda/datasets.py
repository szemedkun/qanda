from __future__ import absolute_import
from __future__ import print_function
from functools import reduce
import re
import numpy as np
np.random.seed(1337)  # for reproducibility
from gensim.models.word2vec import Word2Vec


class Datasets(object):
	def __init__(self,task_index = 1, use10k = False, w2v_dim = 50, only_supporting = False,
		similar_only = False, min_num = 1):
		'''

		'''
		self.only_supporting = only_supporting
		self.use10k = use10k
		self.task_index = task_index
		if self.use10k:
			self.directory = 'data/tasks_1-20_v1-2/en-10k/'
		else:
			self.directory = 'data/tasks_1-20_v1-2/en/'
		self.vocab_size = 0
		self.story_maxlen = 0
		self.query_maxlen = 0
		self.W2V_DIM = w2v_dim
		self.vocab = []
		self.word_idx = {}
		self.glove_dict = None
		self.train = None
		self.test = None
		self.similar_only = similar_only
		self.min_num = min_num


	def _get_task_file(self):
		'''This function is written just for convenience sake. It takes an integer between 1 and 20
		and returns the prefix of the file that has the same task number as the integer from the babi facebook data.
		The default location of the data is where this file is in a folder named data. If that does not exist, one can 
		supply the dir option to the location where the babi dataset is located.

		use10k is by default False. If set to True, the file path to the 10k training and testing cases will be returned

		EXAMPLE: >>>file = get_task_file(3)
				 >>>print(file)
				 >>>data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_
		'''

		task = 'task'+str(self.task_index)
		
		task_dict = {
			'task1': 'qa1_single-supporting-fact_',
			'task2': 'qa2_two-supporting-facts_',
			'task3': 'qa3_three-supporting-facts_',
			'task4': 'qa4_two-arg-relations_',
			'task5': 'qa5_three-arg-relations_',
			'task6': 'qa6_yes-no-questions_',
			'task7': 'qa7_counting_',
			'task8': 'qa8_lists-sets_',
			'task9': 'qa9_simple-negation_',
			'task10': 'qa10_indefinite-knowledge_',
			'task11': 'qa11_basic-coreference_',
			'task12': 'qa12_conjunction_',
			'task13': 'qa13_compound-coreference_',
			'task14': 'qa14_time-reasoning_',
			'task15': 'qa15_basic-deduction_',
			'task16': 'qa16_basic-induction_',
			'task17': 'qa17_positional-reasoning_',
			'task18': 'qa18_size-reasoning_',
			'task19': 'qa19_path-finding_',
			'task20': 'qa20_agents-motivations_'
		}

		return self.directory+task_dict[task]+'{}.txt'

	def _update_word_vec_dict(self):
		'''Updates the word vector dictionary

		'''
		glove_file = 'data/glove_data/glove.6B.'+str(self.W2V_DIM)+'d.txt'
		glove = Word2Vec.load_word2vec_format(glove_file)
		self.glove_dict = glove

	def _update_vocabs(self):
		'''Updates the following attributes in the class
			vocab 
			vocab_size
			story_maxlen
			uery_maxlen
			word_idx

			Must be called after get task file
		'''
		
		challenge = self._get_task_file()
		f = open(challenge.format('train'),'r')
		train = self.get_stories(f)
		self.train = train
		f.close()
		f = open(challenge.format('test'),'r')
		test = self.get_stories(f)
		self.test = test
		f.close()

		vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
		self.vocab = vocab

		vocab_size = len(vocab) + 1
		self.vocab_size = vocab_size
		word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
		self.word_idx = word_idx
		story_maxlen = max(map(len, (x for x, _, _ in train + test)))
		query_maxlen = max(map(len, (x for _, x, _ in train + test)))
		self.story_maxlen = story_maxlen
		self.query_maxlen = query_maxlen

	def fit(self):
		self._update_word_vec_dict()
		self._update_vocabs()


	def tokenize(self, sent):
	    '''Return the tokens of a sentence including punctuation.

	    Adopted from F. Chollet's blog

	    >>> tokenize('Bob dropped the apple. Where is the apple?')
	    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	    '''
	    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

	def parse_stories( self, lines ):
	    '''Parse stories provided in the bAbi tasks format

	    Adopted from Francois Chollet's blog

	    If only_supporting is true, only the sentences that support the answer are kept.
	    '''
	    data = []
	    story = []
	    for line in lines:
	        line = line.decode('utf-8').strip()
	        nid, line = line.split(' ', 1)
	        nid = int(nid)
	        if nid == 1:
	            story = []
	        if '\t' in line:
	            q, a, supporting = line.split('\t')
	            q = self.tokenize(q)
	            substory = None
	            if self.only_supporting:
	                # Only select the related substory
	                supporting = map(int, supporting.split())
	                substory = [story[i - 1] for i in supporting]
	            else:
	                # Provide all the substories
	                substory = [x for x in story if x]
	            data.append((substory, q, a))
	            story.append('')
	        else:
	            sent = self.tokenize(line)
	            story.append(sent)
	    return data


	def get_relevant_sentence(self, stories, question):
		'''This method takes in a list of tokenized sentences and a question and returns min_num of 
		sentences that are relevant or most similar to the question asked for training puposes.
		'''
		similarities = []
		for story in stories:
			story = [s.lower() for s in story]
			question = [q.lower() for q in question]
			similarities.append(self.glove_dict.n_similarity( story, question) )

		relevant_ind = np.array( similarities ).argsort()[::-1][:self.min_num]

		new_stories = []
		for ind in relevant_ind:
			#import pdb; pdb.set_trace()
			new_stories.append( stories[ind] )
		return new_stories

		
	def get_stories(self, f, max_length=None):
	    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

	    Adopted from Francois Chollet's blog

	    If max_length is supplied, any stories longer than max_length tokens will be discarded.
	    '''
	    data = self.parse_stories(f.readlines())
	    
	    if self.similar_only:
	    	new_data = []
	    	for stories, question, answer in data:
	    		new_stories = self.get_relevant_sentence(stories, question)
	    		new_data.append( ( new_stories, question, answer ) )
	    	data = new_data

	    flatten = lambda data: reduce(lambda x, y: x + y, data)
	    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
	    return data

	def get_word_vec(self, word):
		'''Given a dictionary of pretrained word vectors, it returns the vector of the input word

		>>> get_word_vec('the', glove50)
		a numpy array 

		One can download pretrained word vectors from stanford and use gensim to import them

		NOTE: if you are importing glove word vectors in gensim, add the dimensions of the document and the word vec in the first line
		for example add "400000 50" without the quotes if the txt file containes 400000 words and each vector has a dimension of 50.
		'''
		word = word.lower()
		if word in self.glove_dict:
			return self.glove_dict[word]
		else:
			return np.zeros(self.W2V_DIM)

	def vectorize_stories_with_w2v(self, data):
		'''Given a list of tuples with (sentences, question, answer) as data, and a dictionary of word vectors,
		this function returns the vectorized 3D X and Xq and the labels.

		INPUT: data - a list of tuples with (sentences, question, answer) 

		# *** USAGE *** NOt useful any more
		# >>>from gensim.models.word2vec import Word2Vec
		# >>>path = "pathe_to_this_file/glove.6B.50d.txt"
		# >>>glove50 = Word2Vec.load_word2vec_format(path)
		# >>>X, Xq, a = vectorize_stories_with_w2v(data, glove50, vocab_size = 
		# 	36, story_maxlen=25, query_maxlen=5)

		'''

		X = []
		Xq = []
		Y = []
		print("Getting word2vec ...")
		for story, query, answer in data:
			x = np.zeros((self.story_maxlen, self.W2V_DIM))
			xq = np.zeros((self.query_maxlen, self.W2V_DIM))
			xw2v = [self.get_word_vec(w) for w in story]
			xqw2v = [self.get_word_vec(w) for w in query]
			ind_story = self.story_maxlen - len(story)
			ind_q = self.query_maxlen - len(query)

			x[ind_story:, :] = np.array( xw2v )
			xq[ind_q:, :] = np.array( xqw2v )

			y = np.zeros(self.vocab_size)
			y[self.word_idx[answer]] = 1
			X.append(x)
			Xq.append(xq)
			Y.append(y)
		print("Finished getting word2vec ...")
		return np.array(X), np.array(Xq), np.array(Y)

	def get_training_data(self):
		return self.vectorize_stories_with_w2v(self.train)

	def get_testing_data(self):
		return self.vectorize_stories_with_w2v(self.test)



if __name__=="__main__":
	pass
	# model = Datasets()
	# model.fit()
	# X, Xq, Y = model.get_training_data()
	# tX, tXq, tY = model.get_testing_data() 