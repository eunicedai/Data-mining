import sys
import time
import pyspark
from operator import add
from itertools import combinations
from pyspark import SparkContext, SQLContext, StorageLevel, SparkConf
from pyspark.sql import functions
from graphframes import *


threshold, inpath, outpath = int(sys.argv[-3]), sys.argv[-2], sys.argv[-1]

start = time.time()
sc = SparkContext(appName="task1")
sc.setLogLevel('ERROR')

# Data
def load(x):
    arr = x.split(',')
    return (arr[1], arr[0])
users = sc.textFile(inpath).filter(lambda x : x != ('user_id,business_id'))\
        .map(load).groupByKey().mapValues(list)\
        .map(lambda x: sorted(x[1]))\
        .flatMap(lambda x: [(com, 1) for com in combinations(x, 2)])\
        .reduceByKey(add).filter(lambda x: x[1] >= threshold)\
        .flatMap(lambda x: [(x[0][0], x[0][1]), (x[0][1], x[0][0])])
edges = users.collect()
nodes = users.map(lambda x:x[0]).distinct().map(lambda x:(x,x)).collect()

sqlCtx = SQLContext(sc)
vert = sqlCtx.createDataFrame(nodes,['id', 'uid'])
edges = sqlCtx.createDataFrame(edges, ['src', 'dst'])

graph = GraphFrame(vert, edges)
print(graph)

# same community would have same label
community = graph.labelPropagation(maxIter=5)\
                 .groupBy('label')\
                 .agg(functions.collect_list('uid'))\
                 .rdd.map(lambda x: sorted(x[1])).collect()
out = sorted(community, key=lambda x: (len(x), x[0]))
print(out)

with open(outpath, 'w') as f:
    for arr in out:
        f.write("'" + "', '".join(arr) + "'\n")

print("------------Time: ", time.time()-start)
