# Nathan Smith
# UT ID 1000689732
# Programming Lab 2
# 14 April 2016
#
# CSV file parsing adapted from code provided by Kaggle user Kushal Agrawal (decision tree functions not used)
# https://www.kaggle.com/kushal1412/home-depot-product-search-relevance/decision-tree-relevance/code
#
#import beep # The person running this could be using Linux!
import time
from datetime import datetime

print(datetime.now())

from math import log2
from math import log10
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

t = time.time()

node_count = 0

#using 0.2 as a stopping variance
def dtl(train_atts, depth, default):
	global node_count
	global att_names

	node_count += 1
	# print("node %d, %d deep" %(node_count, depth))
	if train_atts.values == []:
		return node(None, default)
	elif depth == 0:
		return node(None, default)
	elif variance(train_atts) <= 0.002: # If the variance gets really small, we do this to keep us sane.
		return node(None, default)
	else: # the condition that does not make a leaf
		tree = node(train_atts) # make a node with the best attribute
		# print(tree.attribute)
		train_left  = train_atts[train_atts[tree.attribute] < tree.thresh]
		train_right = train_atts[train_atts[tree.attribute] >= tree.thresh]#train_atts.columns.values[node.attribute]]
		tree.connect(dtl(train_left, depth-1, sum(train_atts['relevance'])/len(train_atts.values)), dtl(train_right, depth-1, sum(train_atts['relevance'])/len(train_atts.values)))
		return tree
 
# changing a basic thing
# node object (used with the decision tree learner)
class node:
	step = 11
	# branches = []
	value = None
	thresh = None
	attribute = None
	def connect(self, leaf1, leaf2 = None):
		self.branches = [leaf1, leaf2]
		# self.branches.append(leaf1)
		# if leaf2 is not None:
		# 	self.branches.append(leaf2)
	def traverse(self, test, row):
		try:
			self.branches
		except AttributeError:
			if self.value is None:
				print("Warning: Could not classify item (Why are there no parameters?  Hmm...)")
				return None
			else:
				return self.value
		else:
			if self.thresh is None:
				print("This should not happen.")
				return None
			else:
				if test[self.attribute].values[row] < self.thresh: # TODO: return here after testing this function when things load
					return self.branches[0].traverse(test, row)
				else:
					return self.branches[1].traverse(test, row)
	def __init__(self, train_atts = None, val = None):
		# print(self.step)
		if train_atts is None or len(train_atts.values) == 0: #this indicates there's a value
			if val is None:
				print("Warning: Node object created without parameters.")
			else:
				self.value = val
		else:
			max_gain = self.thresh = -1
			global att_names
			for A in att_names:
				att_values = train_atts[A].values
				low = min(att_values)
				high = max(att_values)
				for weight in range(1,self.step):
					curr_thresh = low + int(weight * (high-low)/(self.step))
					gain = infogain(train_atts, A, curr_thresh) # i = attribute number
					# print(max_gain)
					# print(gain)
					if gain > max_gain:
						max_gain = gain
						self.attribute = A
						self.thresh = curr_thresh
			if self.attribute is None:
				print("Attribute not being selected correctly.")

# returns the variance of rankings provided by the system
def variance(train_atts):
	#print(len(train_atts.values))
	if len(train_atts.values) == 0:
		return 0
	total = 0
	relevance = train_atts.relevance.values
	mean = sum(relevance)/len(relevance)
	var = sum([(val-mean)**2 for val in relevance]) # sample, not population
	try:
		return var/(len(relevance)-1)
	except ZeroDivisionError:
		return var/len(relevance)

# returns the 'information gain' of an attribute split based on provided threshold
# (basically just the averages, not an actual info gain formula)
def infogain(train_atts, A, thresh):
	low = train_atts[train_atts[A] < thresh][A]
	high = train_atts[train_atts[A] >= thresh][A]
	return sum(low)/(len(low)+1) + sum(high)/(len(high)+1)

# ====================PARSING==================== #
stemmer = SnowballStemmer('english')

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('data/product_descriptions.csv')

print("%f seconds reading" %(time.time() - t))
#print("This next step will take a while, but it will notify you by audio when it's finished.")
num_train = df_train.shape[0]

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

t = time.time()
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# will perform further operations on single file to save time
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

print("%f seconds stemming" %(time.time() - t))
t = time.time()
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
print("%f seconds finding common words" %(time.time() - t))

df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

t = time.time()
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

# separate out the attribute names into their own array to make parsing easier
att_names = (df_train.drop(['id', 'product_uid', 'relevance'], axis=1)).columns.values
print("%f seconds messing with the arrays" %(time.time() - t))

test_atts = df_test.drop(['id','relevance'],axis=1).values

tree = dtl(df_train, 10, 1)

results = [tree.traverse(df_test, i) for i in range(len(df_test))]

#low = min(results)
#results_norm = [(r-low) for r in results]
#high = max(results_norm)
#results_norm = [2*(r/high) + 1 for r in results_norm]

# classifications = df_train['relevance'].values
# train_atts = df_train.drop(['id','relevance'],axis=1).values

sub = pd.Series(results_norm, index=id_test, name='relevance').to_frame().reset_index()
sub.to_csv('submission.csv', index_label=True, index=False)

#beep.sail() #lets me know when the provided stuff is done

# to convert a number to a name: train_atts.columns.values.tolist()[x] -> name of column x
