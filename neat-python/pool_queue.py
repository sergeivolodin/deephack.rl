import multiprocessing
from time import sleep
from six.moves import cPickle as pickle
from multiprocessing.reduction import reduce_connection
import numpy as np

m = multiprocessing.Manager()
cuda_q = m.Queue()

def compress_pipe(p):
    pp = pickle.dumps(reduce_connection(p))
    return(pp)
def decompress_pipe(pp):
    upw = pickle.loads(pp)
    pp = upw[0](upw[1][0],upw[1][1],upw[1][2])
    return(pp)

def p():
    global cuda_q
    while True:
        [x, p] = cuda_q.get(block = True)
        print("processing " + str(x))
        y = x ** 2
        p = decompress_pipe(p)
        p.send(y)

def worker(name):
    global cuda_q
    a, b = multiprocessing.Pipe()
    x = name
    b = compress_pipe(b)
    cuda_q.put([x, b])
    y = a.recv()
    return([x,y])

pool = multiprocessing.Pool(processes=3)

a = multiprocessing.Process(target = p, args = ())

workers = []
N = 3

a.daemon = True
a.start()

for i in range(N):
    workers.append(pool.apply_async(worker, (i,)))

for i in range(N):
    print(workers[i].get())

a.terminate()
a.join()
