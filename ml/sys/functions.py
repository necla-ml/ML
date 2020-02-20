import sys

def add_path(p):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

def x_available():
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0