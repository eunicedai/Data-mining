import sys
import json
import time
import collections
import random
import itertools
import csv
from pyspark import SparkContext 

inpath, outpath = sys.argv[-2], sys.argv[-1]

sc = SparkContext(appName="task1")
start = time.time()
#data
def load(x):
    ids = x.split(',')
    return (str(ids[0]), str(ids[1]))

lines = sc.textFile(inpath).filter(lambda x: x != ('user_id,business_id,stars'))
# (user, buis)
conn = lines.map(load).distinct() 
# (user: idx)
users = conn.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
# (buis: [user idx])
matrix = conn.map(lambda x: (x[1], users[x[0]])).groupByKey().mapValues(list).map(lambda x: (x[0], sorted(x[1])))
print(matrix.take(5))
print("--- %s len of users ---" % (len(users)))
#map version
dist_m = matrix.collectAsMap()
# matrix = collections.defaultdict(set)
# for u, b in conn.collect():
#     matrix[b].add(users[b])

L_users = len(users)
band = 30
row = 2
hash_fun = []
for _ in range(band):
    a = random.randint(1, 100000)
    b = random.randint(1, 100000)
    p = 24593
    hash_fun.append((a,b,p))
# having bands * matrix
def sigMatrix(nums):
    temp = []
    for (a, b, p) in hash_fun:
        new = []
        for i in nums:
            cal = ((a * i + b) % p)%L_users
            new.append(cal)
        temp.append(min(new)) #first exist user 
    return temp

sigRes = matrix.mapValues(sigMatrix)
'''
buis: [sigs*band]
'''
#LSH
def Divband(nums):
    temp = []
    for i in range(len(nums[1])): 
        new = [nums[1][i]]
        if i%row == 0:
            temp.append(((hash(tuple(new)), i),[nums[0]])) #((band's sig, sig idx), [buis])
    return temp

def genePairs(nums):
    temp = list(itertools.combinations(nums[1], 2))
    res = tuple(sorted(temp))
    return res

def Jaccard(nums):
    ban1, ban2 = nums[0], nums[1]
    user1, user2 = set(dist_m[ban1]), set(dist_m[ban2])
    
    jac = len(user1.intersection(user2))/len(user1.union(user2))
    return ((ban1, ban2), jac)

pair = sigRes.flatMap(Divband).reduceByKey(lambda x, y: x+y)\
            .filter(lambda x : len(x[1]) > 1)\
            .flatMap(lambda x : list(itertools.combinations(x[1],2)))\
            .map(lambda x: tuple(sorted(list(x)))).distinct()
print(pair.take(5))
candi = pair.map(Jaccard).filter(lambda x: x[1]>=0.5).sortByKey()
print(candi.take(5))
with open(outpath, 'w') as csvfile:
    head = ['business_id_1', 'business_id_2', 'similarity']
    writer = csv.writer(csvfile) 
    writer.writerow(head)
    for k, v in candi.collect():
        writer.writerow(list(k)+[v])

print("--- %s seconds ---" % (time.time() - start))