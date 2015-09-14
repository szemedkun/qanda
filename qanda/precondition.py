import numpy as np
from datasets import Datasets
import re
import pickle as pkl

from gensim.models.word2vec import Word2Vec


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    Adopted from smerity's blog

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def get_relevant_sentence( stories, question, glove_dict, threshold = 0.99 ):
	'''This method takes in a list of tokenized sentences and a question and returns min_num of 
	sentences that are relevant or most similar to the question asked for training puposes.
	'''
	with open('stop.pkl', 'rb') as f:
		stop = pkl.load(f)

	similarities = []
	for story in stories:
		story = [s.lower() for s in story if s not in stop]
		question = [q.lower() for q in question]
		sims = []
		for s in story:
			for q in question:
				sims.append( glove_dict.similarity( s, q ) )
		similarities.append( sims )
	#import pdb; pdb.set_trace()

	relevant_ind = sorted( [ind for ind, sim in enumerate( similarities ) 
		if max( sim ) > threshold ] )

	#relevant_ind = sorted( list( np.array( similarities ).argsort()[::-1][:min_num] ) )

	new_stories = []
	#print relevant_ind
	#import pdb; pdb.set_trace()
	for ind in relevant_ind:
		new_stories.append( stories[ind] )
	return new_stories

def get_glove(W2V_DIM = 50):
	glove_file = 'data/glove_data/glove.6B.'+str(W2V_DIM)+'d.txt'
	glove = Word2Vec.load_word2vec_format(glove_file)
	return glove

def get_word_vec(word, glove_dict):
	'''Given a dictionary of pretrained word vectors, it returns the vector of the input word

	>>> get_word_vec('the', glove50)
	a numpy array 

	One can download pretrained word vectors from stanford and use gensim to import them

	NOTE: if you are importing glove word vectors in gensim, add the dimensions of the document and the word vec in the first line
	for example add "400000 50" without the quotes if the txt file containes 400000 words and each vector has a dimension of 50.
	'''
	word = word.lower()
	if word in glove_dict:
		return glove_dict[word]
	else:
		return np.zeros(W2V_DIM)


def precondition(story, question, glove_dict):

	flatten = lambda data: reduce(lambda x, y: x + y, data)

	# Break story into sentences. Tokenize each sentence. Return 2d List [[sent1 - tokenized],[sent2 - tokenized]]
	tokenized_story = tokenize(story)
	tokenized_query = tokenize(question)
	punctuation = ":!.?"
	sent_end = [0] + [i + 1 for i, s in enumerate(tokenized_story) 
	if s in punctuation] + [len(tokenized_story) + 1 ]

	stories = []
	for ind in xrange( len(sent_end) - 2 ):
		sent = tokenized_story[sent_end[ind]:sent_end[ind+1]]
		stories.append( sent )

	new_stories = get_relevant_sentence(stories, tokenized_query, glove_dict)

	return flatten(new_stories), tokenized_query

def vectorize(story, query, glove_dict, story_maxlen = 10, W2V_DIM = 50, query_maxlen = 10):

	x = np.zeros((story_maxlen, W2V_DIM))
	xq = np.zeros((query_maxlen, W2V_DIM))
	xw2v = [get_word_vec(w, glove_dict) for w in story]
	xqw2v = [get_word_vec(w, glove_dict) for w in query]
	ind_story = story_maxlen - len(story)
	ind_q = query_maxlen - len(query)

	x[ind_story:, :] = np.array( xw2v )
	xq[ind_q:, :] = np.array( xqw2v )

	return x, xq


	# Fit word vectors and prepare X, Xq and return them

if __name__ == "__main__":
	glove = get_glove()
	stories, query =  precondition('Bob dropped the apple.', 'Where is the apple?', glove)
	x,qx = vectorize(stories, query, glove)