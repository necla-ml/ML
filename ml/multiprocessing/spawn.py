from torch.multiprocessing.spawn import (
    multiprocessing, 
    spawn, 
    SpawnContext, 
    _python_version_check, 
    _wrap)

import sys

def spawn(fn, args=(), nprocs=1, join=True, daemon=False, context=None):
    if context is None:
        import platform
        system = platform.system()
        if 'Linux' == system:
            context = 'forkserver'

    _python_version_check()
    mp = multiprocessing.get_context(context)
    error_queues = []
    processes = []
    #print(f"mp context={context}, __stdin__: {sys.__stdin__}")
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    spawn_context = SpawnContext(processes, error_queues)
    if not join:
        return spawn_context

    # Loop on join until it returns True or raises an exception.
    while not spawn_context.join():
        pass