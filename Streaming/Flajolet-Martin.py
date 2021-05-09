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
bx = BlackBox()
params = []
h_num = 50
idx = 0
total_g = 0
total_e = 0

def get_prime(x):  
    for possiblePrime in range(x,  x+10000):
        # Assume number is prime until shown it is not. 
        isPrime = True
        for num in range(2, int(possiblePrime ** 0.5) + 1):
            if possiblePrime % num == 0:
                isPrime = False
                break
        if isPrime:
            return possiblePrime

for _ in range(h_num):
    a = random.randint(1,pow(2,20))
    b = random.randint(1,pow(2,20))
    p = get_prime(random.randint(1,pow(2,20)))
    params.append((a,b,p))

def tail_zero(x):
    s = str(x)
    return len(s)-len(s.rstrip('0'))

def myhashs(s):
    #((ax + b) % p) % m
    result = []
    s_int = int(binascii.hexlify(s.encode('utf8')),16)
    for arr in params:
        a, b, p = arr
        out = ((a*s_int + b)%p)%69997
        result.append(out)
    return result

def FM(data):
    global idx
    global total_g
    global total_e

    visited = set()
    est_count = []

    for u in data:
        temp = myhashs(u)
        visited.add(u)
        mxtail = 0
        for h in temp:
            bin_num = bin(h)[2:]
            tail = tail_zero(bin_num)
            mxtail = max(mxtail, tail)
        est_count.append(2**mxtail)
    sub = h_num//3
    group_est = []
    for i in range(3):
        total = sum(est_count[i*sub:(i+1)*sub-1])/sub
        group_est.append(total)
    group_est.sort()
    med = group_est[3//2] 
    
    total_g += len(visited)
    total_e += int(med)

    with open(outpath, 'a') as f:
	    f.write(str(idx)+','+str(len(visited))+','+str(int(med))+'\n')
    idx += 1    

with open(outpath, 'w') as f:
	f.write("Time,Ground Truth,Estimation\n")

for i in range(num_asks):
	data = bx.ask(inpath, s_size)
	FM(data)

print(total_g/total_e)