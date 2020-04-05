class Keeper(object):
    def __init__(self, name=None, suffix=None, backend='tensorboard', **kwargs):
        from datetime import datetime
        from ml import distributed as dist
        now = datetime.now()
        prefix = f"{now.month:02d}{now.day:02d}_{now.hour:02d}-{now.minute:02d}-{now.second:02d}_{dist.hostname()}" if name is None else name
        exp = prefix if suffix is None else f"{prefix}_{suffix}"
        if backend == 'tensorboard':
            self.impl = TensorBoardKeeper(log_dir=f"export/runs/{exp}", **kwargs)
        elif backend == 'visdom':
            self.impl = VisdomKeeper(env=exp, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        return self.impl.add_scalar(tag, scalar_value, global_step, walltime)

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        return self.impl.add_scalars(tag, tag_scalar_dict, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        raise NotImplementedError

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        raise NotImplementedError

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        raise NotImplementedError

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):      
        raise NotImplementedError

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        raise NotImplementedError

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        raise NotImplementedError

    def add_graph(self, model, input_to_model=None, verbose=False):
        raise NotImplementedError

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        raise NotImplementedError

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None):
        raise NotImplementedError

    def add_custom_scalars(self, layout):
        raise NotImplementedError

    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None):
        raise NotImplementedError

    def add_hparams(self, hparam_dict=None, metric_dict=None):
        raise NotImplementedError

    def flush(self):
        self.impl.flush()

    def close(self):
        self.impl.close()

class TensorBoardKeeper(Keeper):
    def __init__(self, 
                 log_dir=None, 
                 purge_step=None, 
                 max_queue=10, 
                 flush_secs=120, 
                 filename_suffix=''):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(
            log_dir=log_dir, 
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
    
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(tag, tag_scalar_dict, global_step, walltime)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

def vis_init(env='exp', clear=True):
    try:
        from visdom import Visdom
    except ImportError:
        raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")
    else:
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

class VisdomKeeper(Keeper):
    def __init__(self, env='main', clear=True, **kwargs):
        self.vis = vis_init(env, clear)
        self.tags = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if tag not in self.tags:
            win = vis_create(self.vis, title=tag, xlabel='step')
            self.tags[tag] = win

        win = self.tags[tag]
        self.vis.line(X=[global_step], Y=[scalar_value], win=win, update='append')

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        if tag not in self.tags:
            legend = list(tag_scalar_dict.keys())
            win = vis_create(self.vis, title=tag, xlabel='step', legend=legend, npts=len(tag_scalar_dict))
            self.tags[tag] = win

        win = self.tags[tag]
        scalars = list(tag_scalar_dict.values)
        self.vis.line(X=[global_step], Y=scalars, win=win, update='append')