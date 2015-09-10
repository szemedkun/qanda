from collections import Counter


def compare_predictions(ds, model, tX, tXq, tY):
	y_pred = model.model.predict_classes( [tX, tXq] )
	y_prob = model.model.predict_proba( [tX, tXq] )
	answers = ds.answers
	test = ds.test

	accumulator = [['Story Len','Actual', 'First_pred', 'Runner_up']]
	for i, ts in enumerate( test ):
		line = [Counter(ts[0])['.'], ts[2], answers[y_pred[i]-1], answers[ y_prob[i,:].argsort()[::-1][1] - 1 ] ]
		accumulator.append( line )
