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

trainpath, testpath, outpath = sys.argv[-3], sys.argv[-2], sys.argv[-1]

sc = SparkContext(appName="task2_1")
start = time.time()

def load1(x):
    ids = x.split(',')
    return ((str(ids[0]), str(ids[1])), float(ids[2]))
def load2(x):
    ids = x.split(',')
    return (str(ids[0]), str(ids[1]))
#data
train = sc.textFile(trainpath).filter(lambda x: x != ('user_id,business_id,stars'))\
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
error = []
def predict(x):
    '''
    p = sum(r*w)/sum(abs(w))
    '''
    u, b = x
    if b not in busi_ur_map: return 3.84
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
    if down == 0: return b_avg_map[b]
    elif mark: return top/down
    else: return ((top/down)+b_avg_map[b])/2

def write(x):
    return ','.join(str(d) for d in x)

pred = test.map(lambda x: (x[0],x[1], predict(x)))
print(pearcorr)
#header = sc.parallelize(["user_id,business_id,prediction"])
#join = header.union(pred)
#join.repartition(1).saveAsTextFile(outpath)
with open(outpath, 'w') as csvfile:
    head = ["user_id","business_id","prediction"]
    writer = csv.writer(csvfile) 
    writer.writerow(head)
    for k in pred.collect():
        writer.writerow(list(k))
    
print("--- %s seconds ---" % (time.time() - start))