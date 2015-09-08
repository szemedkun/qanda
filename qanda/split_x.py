import numpy as np
import datasets as d



def get_indices(doc,max_doc_len):
	ind_list = []
	for ind in xrange( len(doc) ):
		one_row = doc[ind][0]
		const = max_doc_len - len( one_row )
		ind_list_row = [-1] + [ i + const for i in xrange( len( one_row ) ) if one_row[i] == '.' ]
		ind_list.append( ind_list_row )
	return ind_list


def split_x(X, ind_list, num_split = 10, max_sent_len = 10):
	num_obv, num_word, w2v_dim = X.shape
	all_partitions = []
	for ind in xrange( num_obv ):
		ind_list_row = ind_list[ ind ]
		partitions = {}
		for part in xrange( num_split ):
			new_x = np.zeros( ( max_sent_len, w2v_dim ) )
			if part < len( ind_list_row ) - 1:
				ind_max = min( [num_word,ind_list_row[part + 1] + 1 ])
				ind_min = max([ind_list_row[part] + 1, ind_max - max_sent_len] )
				#print 'max',ind_max
				#print 'min',ind_min
				#print num_word
				start_ind = max_sent_len - (ind_max - ind_min)
				new_x[start_ind:, :] = X[ind, ind_min: ind_max, :]
			partitions[part] = new_x
		all_partitions.append( partitions )
	return all_partitions

def make_x(all_parts):
	nb_obs = len( all_parts )
	num_word, w2v_dim = all_parts[0][0].shape
	keys = all_parts[0].keys()
	new_dict = {}
	for key in keys:
		new_dict[key] = np.zeros((nb_obs, num_word, w2v_dim))
		for ind in xrange( nb_obs ):
			new_dict[key][ind, :, : ] = all_parts[ind][key]
	return new_dict

#def reduce_dim(new_dict, max_sent_len = 10):


if __name__ == "__main__":
	#ds = d.Datasets(task_index = 1)
	#ds.fit()
	#X, Xq, Y = ds.get_training_data()
	#train = ds.train
	#max_doc_len = ds.story_maxlen
	ind_list = get_indices( train, max_doc_len )
	all_parts = split_x( X, ind_list )
	new_dict = make_x( all_parts )



	