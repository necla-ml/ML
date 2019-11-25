try:
    import visdom
except ImportError:
    raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")
import numpy as np


def vis_init(env='main', clear=True):
    #vis = visdom.Visdom()
    #print(f"vis.check_connection(): {vis.check_connection()}")
    #if not vis.check_connection():
    #    raise RuntimeError("Visdom server not running. Please run python -m visdom.server")
    vis = visdom.Visdom(env=env)
    clear and vis.close()
    return vis


def vis_create(vis, title='', xlabel='', ylabel='', legend=None, npts=1):
    if npts == 1:
        return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
    else:
        return vis.line(X=np.array([npts * [1]]), Y=np.array([npts * [np.nan]]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend))
