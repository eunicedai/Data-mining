import sys
import time
import pyspark
from operator import add
from collections import defaultdict, deque
from itertools import combinations
from pyspark import SparkContext, SQLContext, StorageLevel, SparkConf
from pyspark.sql import functions
from graphframes import *

threshold, inpath, btw_outpath, comm_outpath = int(sys.argv[-4]), sys.argv[-3], sys.argv[-2], sys.argv[-1]
start = time.time()
sc = SparkContext(appName="task2")
sc.setLogLevel("WARN")

#graph
def load(x):
    arr = x.split(',')
    return (str(arr[1]), str(arr[0]))
data = sc.textFile(inpath).filter(lambda x : x != ('user_id,business_id'))\
        .map(load).groupByKey().mapValues(list)\
        .map(lambda x: sorted(x[1]))\
        .flatMap(lambda x: [(com, 1) for com in combinations(x, 2)])\
        .reduceByKey(add).filter(lambda x: x[1] >= threshold)\
        .flatMap(lambda x: [(x[0][0], x[0][1]), (x[0][1], x[0][0])])
graph = defaultdict(set)
for n1, n2 in data.collect():
    graph[n1].add(n2)
    graph[n2].add(n1)

#Between
def get_btw(node):
    edgeVal = defaultdict(float)
    q = deque([node])
    parent = defaultdict(set)
    parent[node] = set()
    degree = {node: 0}
    node_val = {}
    mx = 0

    while q:
        n = q.popleft()
        for child in graph[n]:
            if child not in parent:
                parent[child].add(n)
                degree[child] = degree[n]+1
                mx = max(mx, degree[child])
                q.append(child)
            else:
                if degree[child] > degree[n]:
                    parent[child].add(n)
                    
    degree = sorted(list(degree.items()), key=lambda x: x[1])
    q = deque([i[0] for i in degree][::-1])
    path = {node: 1}
    for arr in degree[1:]:
        n = arr[0]
        path[n] = sum([path[p] for p in parent[n]])
        
    while q:
        cur = q.popleft()
        for par in parent[cur]:
            if par not in node_val:
                node_val[par] = 1
            val = node_val.get(cur, 1)*path[par]/path[cur]
            node_val[par] += val
            pair = tuple(sorted([cur, par]))
            edgeVal[pair] += val
            
    return edgeVal.items()

nodes = sc.parallelize(graph.keys())
btw = nodes.flatMap(get_btw).reduceByKey(add).map(lambda x: (x[0], x[1]/2))
sort_btw = sorted(btw.collect(), key=lambda x: (-x[1], x[0]))

with open(btw_outpath, 'w') as f:
    for row in sort_btw:
        f.write(str(row[0])+','+str(round(row[1], 5))+'\n')

#Community
def get_community_btw(comm_idx):
    edgeVal = defaultdict(float)
    for node in graph.keys():
        if node not in comm[comm_idx]:
            comm_idx += 1
            comm[comm_idx] = set([node])

        q = deque([node])
        parent = defaultdict(set)
        parent[node] = set()
        degree = {node: 0}
        node_val = {}
        mx = 0

        while q:
            n = q.popleft()
            for child in graph[n]:
                if child not in parent:
                    comm[comm_idx].add(child)
                    parent[child].add(n)
                    degree[child] = degree[n]+1
                    mx = max(mx, degree[child])
                    q.append(child)
                else:
                    if degree[child] > degree[n]:
                        parent[child].add(n)
                        
        degree = sorted(list(degree.items()), key=lambda x: x[1])
        q = deque([i[0] for i in degree][::-1])
        path = {node: 1}
        for arr in degree[1:]:
            n = arr[0]
            path[n] = sum([path[p] for p in parent[n]])
            
        while q:
            cur = q.popleft()
            for par in parent[cur]:
                if par not in node_val:
                    node_val[par] = 1
                val = node_val.get(cur, 1)*path[par]/path[cur]
                node_val[par] += val
                pair = tuple(sorted([cur, par]))
                edgeVal[pair] += val

    return sorted(list(edgeVal.items()), key=lambda x: (-x[1], x[0]))

def del_high_btw():
    if not sort_btw: return False
    pair, high = sort_btw.popleft()
    graph[pair[0]].remove(pair[1])
    graph[pair[1]].remove(pair[0])

    if not sort_btw: return False
    while sort_btw[0][1] == high:
        pair, high = sort_btw.popleft()
        graph[pair[0]].remove(pair[1])
        graph[pair[1]].remove(pair[0])
        if not sort_btw: return False
    return True   

def get_Q():
    def get_sum(pair):
        n1, n2 = pair
        A = 1 if n2 in org_graph[n1] else 0
        return A-(len(graph[n1])*len(graph[n2])/(2*m))

    q_val = sc.parallelize(list(comm.values()))\
               .flatMap(lambda x: [(com, 1) for com in combinations(list(x), 2)])\
               .distinct().map(lambda x: x[0])\
               .map(get_sum).reduce(add)
    
    return q_val

mxQ = -1
res_comm = {}
comm = {0:set()}
m = len(sort_btw)
org_graph = graph
sort_btw = deque(sort_btw)
while del_high_btw():
    sort_btw = deque(get_community_btw(0))
    temp_q = get_Q()/(2*m)
    if temp_q > mxQ:
        mxQ = temp_q
        res_comm = comm
    comm = {0:set()}

del res_comm[0]
res = []
visited = set()
for idx, arr in res_comm.items():
    arr = sorted(arr)
    temp = arr[0]
    if temp not in visited:
        visited.add(temp)
        res.append(arr)
out = sorted(res, key=lambda x: (len(x), x[0]))
with open(comm_outpath, 'w') as f:
    for arr in out:
        f.write("'" + "', '".join(arr) + "'\n")

print("------------Time: ", time.time()-start)