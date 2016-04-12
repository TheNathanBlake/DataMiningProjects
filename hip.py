import winsound #I want to know when it's done!
import time
from math import log2
from math import log10
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

t = time.time()

#using 0.2 as a stopping variance
def dtl(train_atts, depth, default):
	if train_atts.values == []:
		return default
	#elif len(train_atts[0]) == 3: # id, product uid, relevance (with relevance as the class.  We also don't use uid as an att because it can cause overfitting)
		#return (train_atts.values)
	elif depth == 0:
		return train_atts
	else: # the condition that does not make a leaf
		tree = node(train_atts) # make a node with the best attribute
		train_left  = train_atts[train_atts.columns.values[node.attribute]]
		train_right = train_atts[train_atts.columns.values[node.attribute]]
		tree.connect(\
			dtl(train_left, depth-1, sum(train_left.values[2])/len(train_left.values)), \
			dtl(train_right, depth-1, sum(train_right.values[2]/len(train_right.values))))
		return tree
 
# changing a basic thing
# node object (used with the decision tree learner)
class node:
	step = 11
	def connect(self, leaf1, leaf2 = None):
		self.branches.append(leaf1)
		if leaf2 is not None:
			self.branches.append(leaf2)
	def __init__(self, train_atts, ):
		max_gain = self.attribute = self.thresh = -1
		for A in range(2,len(train_atts)):
			att_values = train_atts.values[A]
			low = min(att_values)
			high = max(att_values)
			for weight in range(1,self.step):
				curr_thresh = low + int(weight * (high-low)/(self.step))
				gain = infogain(train_atts, A, thresh) # i = attribute number
				if gain > max_gain:
					max_gain = gain
					self.attribute = A
					self.thresh = curr_thresh

# returns the variance of rankings provided by the system
def variance(train_atts):
	total = 0
	mean = sum(train_atts.values[2])/len(train_atts.values[2])
	var = [(val+mean)**2 for val in train_atts.values[2]]
	return 1/(len(train_atts.values[2])-1)


# returns the information gain of an attribute split based on provided threshold
def infogain(train_atts, A, thresh):
	entropy = 0

# ====================BEGIN=================== #
stemmer = SnowballStemmer('english')

df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../data/attributes.csv')
df_pro_desc = pd.read_csv('../data/product_descriptions.csv')

print("%f seconds reading" %(time.time() - t))

num_train = df_train.shape[0]

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

t = time.time()
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

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

print("%f seconds doing weird things" %(time.time() - t))

classifications = df_train['relevance'].values
train_atts = df_train.drop(['id','relevance'],axis=1).values
test_atts = df_test.drop(['id','relevance'],axis=1).values

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

# to convert a number to a name: train_atts.columns.values.tolist()[x] -> name of column x
