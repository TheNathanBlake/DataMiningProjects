import winsound #I want to know when it's done!
import time
from math import log2
from math import log10
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

t = time.time()

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../data/attributes.csv')
df_pro_desc = pd.read_csv('data/product_descriptions.csv')

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

def dtl(train_atts, depth, default):
	if not train_atts:
		return default
	elif len(train_atts[0]) == 3: # id, product uid, relevance (with relevance as the class.  We also don't use uid as an att because it can cause overfitting)
		return mode(train_atts)
	else:
		best_att = choose(train_atts) # the name of the best attribute, in string form
		tree = node(best_att) # make a node with the best attribute
		best_weight = tree.set_weight(train_atts) # get the best weight for the branch
		for val in train_atts[best_att]: # probably not my best option
			best_atts = # probably a threshold, determined by another function for thresholds for best separation (lowest avg. standard deviation)
		return tree
		
# changing a basic thing
# node object (used with the decision tree learner)
class node:
	#define the threshold percentile step.
	step = 10
	def __init__(self, attribute):
		self.attribute = attribute
		self.branches = []
	def connect(self, leaf):
		self.branches.append(leaf)

def set_weight(values):
	best = [False, False] # [threshold, variance] (looking for lowest variance sum, since high variance in a branch means rankings were less separated)
	low=[]
	high=[]
	for weight in range(1,step+1):
		L = min(values)
		H = max(malues)
		thresh = L + int(weight * (M-L)/(step+1)) # courtesy of the teachings from a particular AI book in the works cited folder
		low = [i in values if i < thresh]
		high = [i in values if i >= thresh]
		low_mean = sum([i in low])/len(low)
		high_mean = sum([i in high])/len(low)
		low_var = sum([(i-low_mean)**2 for i in low]) / (len(low)-1)
		high_var = sum([(i-high_mean)**2 for i in high] ) / (len(high)-1)

		var = low_var + high_var
		if not best[1] or var < best[1]:
			best[0] = weight
			best[1] = var
	self.weight = best[0] #assign the best weight to the node

def choose(train_atts):
	max_gain = best_att = best_thresh = -1
	for i in range(1,len(train_atts)):
	
#rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
#clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)\

# returns the information gain of an attribute split based on provided threshold
def infogain(entities):
	entropy = 0
	