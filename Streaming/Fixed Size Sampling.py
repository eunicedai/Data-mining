import sys
import time
import random
import math
import pyspark
import binascii
from operator import add
from itertools import combinations
from pyspark import SparkContext
from blackbox import BlackBox

inpath, s_size, num_asks, outpath = sys.argv[-4], int(sys.argv[-3]), int(sys.argv[-2]), sys.argv[-1]

random.seed(553)
bx = BlackBox()

mem_size = 100
fix_List = []
count = 0

def reservior(data):
    global mem_size
    global count

    for u in data:
        count += 1
        if mem_size > 0:
            fix_List.append(u)
            mem_size -= 1
        else:
            sel = random.random()
            if sel < 100/count:
                pick = random.randint(0, 99)
                fix_List[pick] = u
        
    if count != 0 and count%100 == 0:
        with open(outpath, 'a') as f:
	        f.write(str(count)+','+str(fix_List[0])+','+str(fix_List[20])+','+str(fix_List[40])+','+str(fix_List[60])+','+str(fix_List[80])+'\n')

with open(outpath, 'w') as f:
	f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")

for i in range(num_asks):
	data = bx.ask(inpath, s_size)
	reservior(data)