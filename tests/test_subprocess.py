from time import time
from tqdm import tqdm
from ml import logging
import subprocess
import pytest

@pytest.fixture
def chunksize():
    return 3

def process_division(args):
    import os
    tid, arg = args
    print(os.getpid(), tid, arg)
    if arg % 2 == 0:
        subprocess.run(['ls'], check=True)
        return tid, True
    else:
        try:
            subprocess.run(['ls', 'dfsdfs'], check=True)
        except Exception as e:
            return tid, e
    
    if True:
        return tid, 1/arg
    else:
        try:
            return tid, 1 / arg
        except Exception as e:
            return tid, e

def test_pool_map(chunksize):
    from multiprocessing import Pool
    import subprocess
    import torch as th
    pool = Pool(processes=3)
    done = []
    failed = []
    t = time()
    tasks = list(range(8))
    args = [1,3,5,0,1,3,0,4]
    chunksize = 16
    for tid, res in tqdm(pool.imap(process_division, zip(tasks, args), chunksize), total=len(tasks)):
        if isinstance(res, Exception):
            e = res
            failed.append(tid)
            logging.error(f"task[{tid}] Failed division due to {e}")
        else:
            done.append(tid)

    logging.info(f"Tasks={list(zip(tasks, args))}")    
    logging.info(f"Done tasks={done}")    
    logging.info(f"Failed tasks={failed}")
    assert len(done) == 3
    assert len(failed) == len(tasks) - len(done)