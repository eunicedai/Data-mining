import sys
import pandas as pd
import numpy as np
import json
import time
import collections
import math
from pyspark import SparkContext
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
#from sklearn.model_selection import GridSearchCV

start = time.time()
folder, testpath, outpath = sys.argv[-3], sys.argv[-2], sys.argv[-1]
sc = SparkContext(appName="task2_2")

train = pd.read_csv(folder+'/yelp_train.csv')
b_file = folder+'/business.json'
busi = sc.textFile(b_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'], x['review_count'],x['stars'],x['attributes'])).collect()
b_df = pd.DataFrame(busi, columns=['business_id','b_review_count','b_star','attributes'])

'''
Clean business Data: pick attribute, top 100 cate
'''
#attr
attr = pd.DataFrame(b_df['attributes'].to_dict()).T
attr = attr.drop(columns=['BusinessParking', 'Ambience', 'BestNights', 'GoodForMeal', 'HairSpecializesIn', 'Music'])
cols_to_split = ['AgesAllowed', 'Alcohol', 'BYOBCorkage', 'NoiseLevel', 'RestaurantsAttire', 'Smoking', 'WiFi']
new_cat = pd.concat([pd.get_dummies(attr[col], prefix=col, prefix_sep='_') for col in cols_to_split], axis=1)
attr = pd.concat([attr, new_cat], axis=1)
attr.drop(cols_to_split, inplace=True, axis=1)
del(new_cat)
attr2 = attr
attr2 = attr2.fillna(0.5).applymap(lambda x: 1 if x != 0.5 else x).applymap(lambda x: 0 if x == 'False' else x)
b_df2 = b_df.merge(attr2, left_index=True, right_index=True)
b_df2 = b_df2.drop(columns=['attributes'])
b_df2.set_index('business_id', inplace=True)
b_df2.head()
print('-----------',b_df2.isnull().sum(),'-------------')
#category
#cate = list(b_df.categories.str.replace(" ", ""))
#cate = pd.Series(list(dum))
#cate_df = pd.get_dummies(cate, sparse=True)
#top = cate_df.astype(bool).sum(axis=0)
#top = top.sort_values(ascending=False)[:100]
#cate_df = cate_df[top.axes[0]]
#merge together
#b_df2 = b_df2.merge(cate_df, left_index=True, right_index=True)


'''
train and business data combine
'''
group = train
group.set_index(['user_id', 'business_id'], inplace=True)
group.sort_index(inplace=True)
combine = pd.merge(group, b_df2, left_index=True, right_index=True)
combine.head()
del(train)

'''
Clean user data
'''
#user_id	u_review_count	useful	funny	cool	fans	u_star
u_file = folder+'/user.json'
user = sc.textFile(u_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['user_id'], x['review_count'],x['useful'],x['funny'],x['cool'],x['fans'],x['average_stars'])).collect()
u_df = pd.DataFrame(user, columns=['user_id','u_review_count','useful','funny','cool','fans','u_star'])
print(u_df.head())
del(u_file)

u_df2 = u_df
u_df2.set_index('user_id', inplace=True)
'''
combine all data together
'''
com2 = combine
com2 = pd.merge(com2, u_df2, how='left', left_index=True, right_index=True)
print('--------com',com2.isnull().sum(),'-------------')


'''
Prediction
'''
test = pd.read_csv(testpath)
test=test.drop(columns=['stars'])
test.set_index(['user_id', 'business_id'], inplace=True)
test.sort_index(inplace=True)
#preprocess test data
test2 = pd.merge(test, b_df2, how='left', left_index=True, right_index=True)
test2 = pd.merge(test2, u_df2, how='left', left_index=True, right_index=True)

y = com2.stars
X = com2.drop(columns=['stars'])
t_Xtest = test2
#tune and find the best para
# param_grid = [{'learning_rate': [0.001,0.01,0.1],
#                'max_depth': [3, 4, 5], 
#                'subsample': [0.6, 0.8, 1.0]}]

# grid_xgb2 = GridSearchCV(XGBRegressor(), param_grid) 
# grid_xgb2.fit(X, y) 
# grid_xgb2.best_params_
#max_depth=5, subsample=0.8, lr=0.1

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=1) 
xgb.fit(X, y)
y_xgb= xgb.predict(t_Xtest)
test['prediction']=y_xgb
test = test.reset_index()
test.to_csv(outpath, index=False)

print('------------',time.time()-start,'--------------')
