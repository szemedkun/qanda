import precondition as p
import pickle as pkl

def predict(pickled_model, story, query):
	result = {}
	
	with open(pickled_model, 'rb') as f:
		model = pkl.load(f)
	X = model.X
	Xq = model.Xq
	
	glove = p.get_glove()
	stories, query =  p.precondition(story, query, glove)
	x,qx = p.vectorize(stories, query, glove, story_maxlen = model.X.shape[1], 
	 query_maxlen = model.Xq.shape[1])

	answers = model.answers

	y_pred = model.model.predict_classes( [x, qx] )
	y_prob = model.model.predict_proba([x,qx])

	for i, answer in enumerate( answers ):
		result[ answer ] = y_prob[i+1]


	return result