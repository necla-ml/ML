# FIXME PyTorch may fail to access CUDA information after daemonization
# See #23401 for details and #29101 for a possible fix


def is_available():
    import ctypes
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            del cuda
            try:
                import torch.cuda
            except ImportError:
                continue
            return True
    else:
        return False


def device_count():
    import os
    key = 'CUDA_VISIBLE_DEVICES'
    if key in os.environ:
        return len(os.environ[key].split(','))
    else:
        import subprocess
        output = subprocess.getoutput("python -c 'import torch as th; print(th.cuda.device_count())'")
        return int(output) if output.isnumeric() else 0