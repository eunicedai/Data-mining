import sys
import json
import time
import collections
import heapq
import random
import itertools
import csv
import math
import pandas as pd
from pyspark import SparkContext 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

folder, testpath, outpath = sys.argv[-3], sys.argv[-2], sys.argv[-1]

sc = SparkContext(appName="task2_3")
start = time.time()
'''
item_base
'''
def load1(x):
    ids = x.split(',')
    return ((str(ids[0]), str(ids[1])), float(ids[2]))
def load2(x):
    ids = x.split(',')
    return (str(ids[0]), str(ids[1]))
#data
train = sc.textFile(folder+'/yelp_train.csv').filter(lambda x: x != ('user_id,business_id,stars'))\
            .map(load1).groupByKey().mapValues(lambda x: sum(x)/len(x))
test = sc.textFile(testpath).filter(lambda x: x != ('user_id,business_id,stars'))\
            .map(load2)
#map with data
busi_u_map = train.map(lambda x: (x[0][1], x[0][0]))\
            .groupByKey().mapValues(set).collectAsMap()
user_b_map = train.map(lambda x: (x[0][0], x[0][1]))\
            .groupByKey().mapValues(set).collectAsMap()
busi_ur_map = train.map(lambda x: (x[0][1], (x[0][0], x[1])))\
            .groupByKey().mapValues(dict).collectAsMap()
user_br_map = train.map(lambda x:(x[0][0], (x[0][1], x[1])))\
            .groupByKey().mapValues(dict).collectAsMap()
b_avg_map = train.map(lambda x:(x[0][1], (x[1], 1)))\
            .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))\
            .map(lambda x: (x[0], x[1][0]/x[1][1])).collectAsMap()

pearcorr = {}
#pearson correlation
def pear_corr(x):
    b1, b2 = x
    sameU = busi_u_map[b1] & busi_u_map[b2]
    L = len(sameU)
    if L == 0: return 0
    star1 = [busi_ur_map[b1][u] for u in sameU]
    star2 = [busi_ur_map[b2][u] for u in sameU]
    avg1 = sum(star1)/len(star1)
    avg2 = sum(star2)/len(star2)
    top1 = [(i-avg1) for i in star1]
    top2 = [(i-avg2) for i in star2]
    top = sum([(a*b) for a, b in zip(top1, top2)])
    down = math.sqrt(sum([i**2 for i in top1]))*math.sqrt(sum([j**2 for j in top2]))
    res = top/down if down != 0 else 0
    #(b1, b2, corrs)
    return res

neigN = 30
#predict
def item_predict(x):
    '''
    p = sum(r*w)/sum(abs(w))
    '''
    u, b = x
    if b not in busi_ur_map: return (3.84, 0)
    otherb = user_b_map[u]
    heap = []
    top, down = 0, 0
    mark = False
    for ob in otherb:
        if (ob, b) not in pearcorr:
            ans = pear_corr((b, ob))
            pearcorr[(b, ob)] = ans
            pearcorr[(ob, b)] = ans
        w = pearcorr[(b,ob)]
        heap.append((w, user_br_map[u][ob]))
    if len(heap) < 10:
        for w, r in heap:
            if w < 0: 
                w = -w
                w = w**2.5
                top += ((6-r)*w)
                down += w
            elif w > 0:
                w = w**2.5
                top += (r*w)
                down += w
    else:
        for w, r in heap:
            if w > 0: 
                w = w**2.5
                top += (r*w)
                down += w
    ans = 0
    if down == 0: ans = b_avg_map[b]
    elif mark: ans = top/down
    else: ans = ((top/down)+b_avg_map[b])/2
    return (ans, len(heap))

pred_map = test.map(lambda x: ((x[0],x[1]), item_predict(x))).collectAsMap()


'''
model_base
'''
#data
train_df = pd.read_csv(folder+'/yelp_train.csv')
b_file = folder+'/business.json'
busi = sc.textFile(b_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'], x['review_count'],x['stars'],x['attributes'])).collect()
b_df = pd.DataFrame(busi, columns=['business_id','b_review_count','b_star','attributes'])

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

group = train_df
group.set_index(['user_id', 'business_id'], inplace=True)
group.sort_index(inplace=True)

combine = pd.merge(group, b_df2, left_index=True, right_index=True)
combine.head()
del(train_df)

u_file = folder+'/user.json'
user = sc.textFile(u_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['user_id'], x['review_count'],x['useful'],x['funny'],x['cool'],x['fans'],x['average_stars'])).collect()
u_df = pd.DataFrame(user, columns=['user_id','u_review_count','useful','funny','cool','fans','u_star'])
del(u_file)
u_df2 = u_df
u_df2.set_index('user_id', inplace=True)

com2 = combine
com2 = pd.merge(com2, u_df2, how='left', left_index=True, right_index=True)

test_df = pd.read_csv(testpath)
test_df=test_df.drop(columns=['stars'])
test_df.set_index(['user_id', 'business_id'], inplace=True)
test_df.sort_index(inplace=True)
#preprocess test data
test2 = pd.merge(test_df, b_df2, how='left', left_index=True, right_index=True)
test2 = pd.merge(test2, u_df2, how='left', left_index=True, right_index=True)

y = com2.stars
X = com2.drop(columns=['stars'])
t_Xtest = test2

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=1) 
xgb.fit(X, y)
y_xgb= xgb.predict(t_Xtest)

def together_predict(x):
    u, b = x
    ansT, L = item_predict(x)
    queryD = test2.query('user_id== @u and business_id == @b')
    ansM = xgb.predict(queryD)[0]
    out = 0
    if L > 10: out = 0.5*ansT+0.5*ansM
    elif L > 5: out = 0.4*ansT+0.6*ansM
    elif L > 3: out = 0.2*ansT+0.8*ansM
    elif L > 2: out = 0.1*ansT+0.9*ansM
    else: out = ansM
    return out

def write(x):
    return ','.join(str(d) for d in x)

'''
hybrid
'''
test_df['prediction']=y_xgb
test_df = test_df.reset_index()

test_df['prediction'] = test_df.apply(lambda x: 0.2*pred_map[(x.user_id, x.business_id)][0]+0.8*x.prediction if pred_map[(x.user_id, x.business_id)][1] > 20 else 0.99*x.prediction+0.01*pred_map[(x.user_id, x.business_id)][0], axis = 1)

test_df.to_csv(outpath, index=False)
#header = sc.parallelize(["user_id,business_id,prediction"])
#join = header.union(pred)
#join.repartition(1).saveAsTextFile(outpath)


    
print("--- %s seconds ---" % (time.time() - start))