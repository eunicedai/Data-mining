import sys
import json
import csv
import time
import collections
import itertools
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *

threshold, sup, inpath, outpath = int(sys.argv[-4]), int(sys.argv[-3]), sys.argv[-2], sys.argv[-1]
start = time.time()

#function
def Apass1(baskets):
    a = baskets[0]
    for basket in baskets[1:]:
        a = a|basket
    return list(set(a))

def getFreq(candi, baskets, sup2):
    count = collections.defaultdict(int)
    for b in baskets:
        for c in candi:
            if set(c).issubset(b):
                count[tuple(sorted(c))] += 1
    res = [k for k, v in count.items() if v >= sup2]
    return res

def generator(k, arr):
    res = []
    n = len(arr)
    for i in range(n-1):
        for j in range(i+1, n):
            if len(arr[i] | arr[j]) == k:
                res.append(tuple(sorted(arr[i] | arr[j])))
    return res                

def getCandi(freq):
    res = set()
    for S in freq:
        for k in S:
            if k not in res: res.add(k)
    return list(res)

def Apriori(info):
    info = [set(i) for i in list(info)]
    chunk = len(info)
    sup2 = sup*chunk/B_count
    
    totalB = Apass1(info)
    cur_candi = [{i} for i in totalB]
    freq_candi = []
    #pass1
    tempF = getFreq(cur_candi, info, sup2)
    freq_candi += tempF
    cur_candi = [{i[0]} for i in tempF]

    L = 2
    while cur_candi:
        pairs = generator(L, cur_candi)
        if not pairs: break
        pairs = [set(i) for i in set(pairs)]
        tempF = getFreq(pairs, info, sup2)
        if not tempF: break
        cur_candi = [set(i) for i in tempF]
        freq_candi += cur_candi
        L += 1
    
    return [tuple(sorted(list(i))) for i in freq_candi]

def totalCount(each, freq):
    each = list(each)
    count = collections.defaultdict(int)
    for e in each:
        for f in freq:
            if set(f).issubset(set(e)): count[f] += 1
    return list(count.items())

#preprocessing
sc = SparkContext(appName="task2")
sqlContext = SQLContext(sc)
df = sqlContext.read.format('csv') \
        .option('header',True) \
        .option('multiLine', True) \
        .load(inpath)
        
test = df.rdd.map(lambda row: (str(row.TRANSACTION_DT[:-4]+row.TRANSACTION_DT[-2:]+"-"+row.CUSTOMER_ID), str(int(row.PRODUCT_ID))))
# test = sqlContext.createDataFrame(test.collect(), StructType([StructField("DATE-CUSTOMER_ID", StringType(), True), 
#                                                               StructField("PRODUCT_ID", StringType(), True)]))

# test.write.csv('Customer_product.csv')

#read csv
bask = test.groupByKey().filter(lambda x: len(x[1]) > threshold)\
                        .map(lambda x: list(x[1]))
B_count = bask.count()

#MapReduce 1
son1 = bask.mapPartitions(Apriori).map(lambda x:(x,1)) \
            .groupByKey().map(lambda x: x[0]) \
            .collect()
print("son1----------")
candiP = sorted(son1, key=lambda x:(len(x),x))
print(candiP)
#MapReduce 2
son2 = bask.mapPartitions(lambda x: totalCount(x, candiP)) \
            .reduceByKey(lambda x,y: x+y).filter(lambda x: x[1]>=sup).keys()
freqP = sorted(sorted(son2.collect()), key=lambda x:(len(x),x))

with open(outpath,'w') as f:
    f.write('Candidates:\n')
    out = ""
    L = 1
    for c in candiP:
        if len(c) == L:
            out += str(c).replace(",)",")") + ','
        else:
            out = out[:-1]+'\n\n'
            L = len(c)
            out += str(c) + ','     
    f.write(out[:-1]+'\n')
    f.write('\nFrequent Itemsets:\n')
    L = 1
    out = ""
    for c in freqP:
        if len(c) == L:
            out += str(c).replace(",)",")") + ','
        else:
            out = out[:-1]+'\n\n'
            L = len(c)
            out += str(c) + ',' 
    f.write(out[:-1])

print("Duration: %s" %(time.time()-start))