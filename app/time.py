import time
from better_profanity import profanity
import os

import multiprocessing

def task(xx):
    return[profanity.censor(y) for y in xx]

x = ['i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.i am sam fuck.']*1000

x1 = x[0:20]
x2 = x[20:40]
x3 = x[40:60]
x4 = x[60:80]

def multiprocessing_function():
    aaa = time.time()
    pool = multiprocessing.Pool(processes = 4)
    res = pool.map(task,(x1,x2,x3,x4))
    pool.close()
    pool.join()
    bbb = time.time()
    print(bbb-aaa)

if __name__ == "__main__":
    multiprocessing_function()