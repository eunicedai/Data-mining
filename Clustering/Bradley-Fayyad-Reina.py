import csv
import json
import time
import random
import numpy as np
from collections import defaultdict
from pyspark import SparkContext
from sklearn.cluster import KMeans

start = time.time()
inpath, n_cluster, outpath = sys.argv[-3], int(sys.argv[-2]), sys.argv[-1]

#Read Data
points = []
point_pos = {}
with open(inpath, 'r') as f:
    for line in f.readlines():
        line = line.split(',')
        fs = []
        for x in line[2:]:
            fs.append(float(x))
        point_pos[int(line[0])] = fs
        points.append((int(line[0]), line[1], fs))
# shuffle and split data for each 20%
random.shuffle(points)
chunks = np.array_split(points, 5)
dimension = len(point_pos[0])

#init data set
RS, DS, CS, DS_c, CS_c = set(), [], [], defaultdict(dict), defaultdict(dict)

def start(chunk):
    global RS
    global DS
    global CS
    global DS_c
    global CS_c
    #basic setup
    feature, idx = [], []
    for i, c, arr in chunk:
        feature.append(arr)
        idx.append(i)
        
    #step2
    Kdata = KMeans(n_clusters = n_cluster*5).fit(feature)
    
    #step3-to RS
    cluster = defaultdict(list)
    for i, l in zip(idx, Kdata.labels_):
        cluster[l].append(i)
    d_feat = set()
    for l, v in cluster.items():
        if len(v) == 1:
            RS.add(v[0])
        else:
            for j in v:
                d_feat.add(j)
                
    #step4
    df, idx = [], [] 
    for d in d_feat:
        df.append(point_pos[d])
        idx.append(d)
    Kdata = KMeans(n_clusters = n_cluster).fit(df)
    
    #step5
    DS = Kdata.cluster_centers_
    for i, l in zip(idx, Kdata.labels_):
        if 'S' not in DS_c[l]: 
            DS_c[l]['P'] = [i]
            DS_c[l]['N'] = 0
            DS_c[l]['S'] = point_pos[i]
            DS_c[l]['SQ'] = point_pos[i]
        else:
            DS_c[l]['P'].append(i)
            DS_c[l]['N'] += 1
            DS_c[l]['S'] = np.sum([DS_c[l]['S'], point_pos[i]], axis = 0)
            DS_c[l]['SQ'] = np.sum([DS_c[l]['SQ'], [p**2 for p in point_pos[i]]], axis = 0)
    
    #step6
    feat2, idx = [], []
    for i in RS:
        feat2.append(point_pos[i])
        idx.append(i)
    L_RS = len(RS)
    Kdata = KMeans(n_clusters = int(1+L_RS/2)).fit(feat2)
    temp_c = defaultdict(list)
    for i, l in zip(idx, Kdata.labels_):
        temp_c[l].append(i)
    RS = set()
    for l, p in temp_c.items():
        if len(p) == 1:
            RS.add(p[0])
        else:
            CS.append(p)
    for i, p in enumerate(CS):
        for pos in p:
            if 'S' not in CS_c[i]:
                CS_c[i]['P'] = [pos]
                CS_c[i]['N'] = 0
                CS_c[i]['S'] = point_pos[pos]
                CS_c[i]['SQ'] = point_pos[pos]
            else:
                CS_c[i]['P'].append(pos)
                CS_c[i]['N'] += 1
                CS_c[i]['S'] = np.sum([CS_c[i]['S'], point_pos[pos]], axis = 0)
                CS_c[i]['SQ'] = np.sum([CS_c[i]['SQ'], [v**2 for v in point_pos[pos]]], axis = 0)

def write_round(f, times):
    ds_count = 0
    cs_clu = 0
    cs_count = 0
    for key in DS_c.keys(): ds_count += DS_c[key]['N']
    for key in CS_c.keys():
        cs_clu += 1
        cs_points_count += DS_c[key]['N']
    rs_count = len(RS)
    f.write("Round " + str(times) + ": " + str(ds_count) + "," + str(cs_clu) + "," + str(cs_count) + "," + str(rs_count) + "\n")

start(chunks[0])
with open(outpath, 'w') as f:
	f.write("The intermediate results:\n")
    write_round(f, 1)
    
'''
Function
'''
def mah_distance_ds(p1, cen_label):
    cen = DS_c[cen_label]['S']/DS_c[cen_label]['N']
    var = np.sqrt((DS_c[cen_label]['SQ']/DS_c[cen_label]['N'])-(cen**2))
    return np.sqrt(sum(((p1-cen)/var)**2))

def mah_distance_cs(p1, cen_label):
    cen = CS_c[cen_label]['S']/CS_c[cen_label]['N']
    var = np.sqrt((CS_c[cen_label]['SQ']/CS_c[cen_label]['N'])-(cen**2))
    return np.sqrt(sum(((p1-cen)/var)**2))

dis_compare = 2*np.sqrt(dimension)

def get_close(point, L_clu, who):
    close = float('inf')
    cluster = -1
    for l in range(L_clu):
        if who == 'DS':
            dis = mah_distance_ds(point, l)
        else:
            dis = mah_distance_cs(point, l)
        if dis < dis_compare:
            if dis < close:
                close = dis
                cluster = l
    return cluster

def bfr(chunk):
    global RS
    global DS
    global CS
    global DS_c
    global CS_c
    feature, idx = [], []
    for i, c, arr in chunk:
        feature.append(arr)
        idx.append(i)
    #step 8
    not_assign = set()
    for i, p in zip(idx, feature):
        close = get_close(p, n_cluster, 'DS')
        if close != -1:
            DS_c[close]['P'].append(i)
            DS_c[close]['N'] += 1
            DS_c[close]['S'] = np.sum([DS_c[close]['S'], p], axis = 0)
            DS_c[close]['SQ'] = np.sum([DS_c[close]['SQ'], [v**2 for v in p]], axis = 0)
        else:
            not_assign.add(i)
    #renew_ds()
    
    #step 9, 10
    L_cs = len(CS)
    for i in not_assign:
        p = point_pos[i]
        close = get_close(p, L_cs, 'CS')
        if close != -1:
            CS_c[close]['N'] += 1
            CS_c[close]['P'].append(i)
            CS_c[close]['S'] = np.sum([CS_c[close]['S'], p], axis = 0)
            CS_c[close]['SQ'] = np.sum([CS_c[close]['SQ'], [v**2 for v in p]], axis = 0)
        else:
            RS.add(i)
    #renew_cs()
    
    #step 11
    feat2, idx = [], []
    for i in RS:
        feat2.append(point_pos[i])
        idx.append(i)
    L_RS = len(RS)
    Kdata = KMeans(n_clusters = int(1+L_RS/2)).fit(feat2)
    temp_c = defaultdict(list)
    for i, l in zip(idx, Kdata.labels_):
        temp_c[l].append(i)
    RS, temp_cs = set(), []
    for l, p in temp_c.items():
        if len(p) == 1:
            RS.add(p[0])
        else:
            temp_cs.extend(p)
    
    #step 12
    L_cs = len(CS)
    for p in temp_cs:
        close = get_close(point_pos[p], L_cs, 'CS')
        if close != -1:
            CS_c[close]['N'] += 1
            CS_c[close]['P'].append(i)
            CS_c[close]['S'] = np.sum([CS_c[i]['S'], point_pos[p]], axis = 0)
            CS_c[close]['SQ'] = np.sum([CS_c[close]['SQ'], [v**2 for v in point_pos[p]]], axis = 0)
        else:
            RS.add(i)

for i in range(1, 5):
    bfr(chunks[i])
    write_round(f, i+1)

res = {}
for group in DS_c.keys():
    for index in DS_c[group]['P']: res[index] = group
for group in CS_c.keys():
    for index in CS_c[group]['P']: res[index] = -1
for index in RS:
    res[index] = -1

f.write("\nThe clustering results: ")
for index, group in sorted(res.items(), key=lambda x: x[0]):
    f.write("\n" + str(index) + "," + str(group))

print("TIME:------------", start-time.time())