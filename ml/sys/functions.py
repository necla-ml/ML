import sys

def add_path(p):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)