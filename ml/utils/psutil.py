def cpu_count(logical=True):
    try:
        import psutil
        return psutil.cpu_count(logical)
    except ImportError:
        import os
        return os.cpu_count() if logical else os.cpu_count() // 2

def get_num_threads():
    return cpu_count(False)