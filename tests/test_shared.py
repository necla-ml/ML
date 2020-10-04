import sys
import time

import torch
from torch.multiprocessing import Process 
from torch.multiprocessing import Queue, SimpleQueue
from torch.multiprocessing import JoinableQueue

#q = SimpleQueue() 
#q = Queue() 
q = JoinableQueue() 

def torch_shared_mem_process(shared_memory):
    counter = 0
    start = time.time()
    while True:
        data = q.get()
        counter += 1
        if data is None:
            print(f'[torch_shared_mem_process_q1] Received with shared memory {shared_memory}: {time.time() - start}')
            return
        # assert data.is_shared()
        del data

def test_mem_share(share_memory):
    p = Process(target=torch_shared_mem_process, args=(share_memory, ))
    p.start()

    start = time.time()
    n = 100
    for i in range(n):
        data = torch.zeros([5, 1280, 720, 3], dtype=torch.float, pin_memory=True)
        if share_memory:
            data.share_memory_()
            q.put(data)
        else:
            q.put(data.numpy())
    q.put(None)
    p.join()
    return time.time() - start

def test_share_mem():
    print()
    with_shared_memory = test_mem_share(share_memory=True)
    no_shared_memory = test_mem_share(share_memory=False)
    print(f'Took {no_shared_memory:.1f} s without shared memory.')
    print(f'Took {with_shared_memory:.1f} s with shared memory.')
    with_shared_memory = test_mem_share(share_memory=True)
    no_shared_memory = test_mem_share(share_memory=False)
    print(f'Took {no_shared_memory:.1f} s without shared memory.')
    print(f'Took {with_shared_memory:.1f} s with shared memory.')
    with_shared_memory = test_mem_share(share_memory=True)
    no_shared_memory = test_mem_share(share_memory=False)
    print(f'Took {no_shared_memory:.1f} s without shared memory.')
    print(f'Took {with_shared_memory:.1f} s with shared memory.')

def test_share_memory():
    print()
    for i in range(100):
        t = torch.empty(5, 3, 720, 1280)
        t.share_memory_()
        q.put(t)