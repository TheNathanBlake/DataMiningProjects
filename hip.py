import time
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
	elif len(train_atts[0]) == 1:
		return mode(train_atts)
	else:
		best = choose(train_atts) # the name of the best attribute, in string form
		tree = #decision tree with root test best
		for val in best:


class node:
	def __init__(self, attribute):
		self.attribute = attribute
		self.branches = []
	def give_weight(self, weight):
		self.weight = weight
	def connect(self, leaf):
		self.branches.append(leaf)

#rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
#clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)