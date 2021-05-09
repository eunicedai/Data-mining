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

# stream_users = bx.ask(file_name, stream_size)

inpath, s_size, num_asks, outpath = sys.argv[-4], int(sys.argv[-3]), int(sys.argv[-2]), sys.argv[-1]

start = time.time()
bx = BlackBox()

bit_array = [0]*69997
params = []
h_num = 30
cur_ID = set()
FP = 0.0
TN = 0.0
idx = 0

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

def myhashs(s):
    #((ax + b) % p) % m
    result = []
    s_int = int(binascii.hexlify(s.encode('utf8')),16)
    for arr in params:
        a, b, p = arr
        out = ((a*s_int + b)%p)%69997
        result.append(out)
    return result

def bloomF(stream_users):
    global idx
    global FP
    global TN
    global h_num

    for u in stream_users:
        temp = myhashs(u)
        hash_num = 0
        for i in range(h_num):
            if bit_array[temp[i]] == 1:
                hash_num += 1
        if u not in cur_ID:
            if hash_num == h_num:
                FP += 1
            else: TN += 1
        cur_ID.add(u)

        for i in range(h_num):
            bit_array[temp[i]] = 1
    rate = FP / (FP + TN)

    with open(outpath, 'a') as f:
	    f.write(str(idx)+','+str(rate)+'\n')
    
    idx += 1
        

with open(outpath, 'w') as f:
	f.write("Time,FPR\n")
    
for i in range(num_asks):
	data = bx.ask(inpath, s_size)
	bloomF(data)
