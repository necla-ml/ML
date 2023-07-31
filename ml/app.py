import os, sys, random
import subprocess
from pathlib import Path

import ml
from ml import (
    cuda,
    distributed as dist,
    multiprocessing as mp,
    utils,
    logging,)

def init_cuda(cfg):
    if cfg.no_gpu:
        # No use of GPU
        cfg.gpu = []
        os.environ['CUDA_VISIBLE_DEVICES'] = 'NoDevFiles'
    else:
        if cfg.gpu is None:
            # Set CUDA Visible GPUs if any
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] != 'NoDevFiles':
                cfg.gpu = sorted(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
            else:
                cfg.gpu = list(range(cuda.device_count()))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(sorted(map(str, cfg.gpu)))
        
def init(cfg):
    init_cuda(cfg)
    random.seed(cfg.seed, deterministic=cfg.deterministic)
    if (cfg.logging or cfg.daemon) and cfg.logfile is None:
        from __main__ import __file__ as script
        cfg.logging = True
        name = Path(script).stem
        if cfg.rank < 0:
            cfg.logfile = f"{name}-{os.getpid()}.log"
        else:
            cfg.logfile = f"{name}-{os.getpid()}_{cfg.rank}.log"

    if cfg.logfile:
        logging.info(f"Logging to {cfg.logfile}")
        logging.basicConfig(filename=cfg.logfile, rank=cfg.rank, world_size=cfg.world_size)
    else:
        logging.basicConfig(stream=sys.stdout, rank=cfg.rank, world_size=cfg.world_size)

    if cfg.dist:
        if cfg.world_size > 1:
            dist.init_process_group(init_method=cfg.dist_url, backend=cfg.dist_backend, rank=cfg.rank, world_size=cfg.world_size)
            logging.info(f"[{cfg.rank}/{cfg.world_size}] '{dist.hostname()}' distributed with {cfg.dist_backend} using {cfg.dist_url}")
            for key in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES']:
                if key in os.environ:
                    logging.info(f"[{cfg.rank}/{cfg.world_size}] {key}: {os.environ[key]}")
        else:
            logging.info(f"HOST: {dist.hostname()} w/o distributed communication")

def exec(main, cfg, *args, **kwargs):
    if cfg.daemon:
        if sys.stdin and sys.__stdin__ and sys.__stdin__.closed:
            sys.__stdin__ = sys.stdin

        from daemon import daemon
        with daemon.DaemonContext(umask=0o022, 
                                chroot_directory=None, 
                                working_directory=os.getcwd(),
                                stdout=sys.stdout, 
                                stderr=sys.stderr,
                                ) as ctx:
            init(cfg)
            with open(cfg.logfile, 'a') as log:
                # XXX redirect stdout in case of existing print()
                daemon.redirect_stream(ctx.stdout, log)
                daemon.redirect_stream(ctx.stderr, log)
                main(cfg, *args, **kwargs)
    else:
        init(cfg)
        main(cfg, *args, **kwargs)

def launch(rank, main, cfg, args, kwargs):
    # New GPU worker proc
    assert rank >= 0
    assert cfg.world_size > 0
    assert cfg.dist
    cfg.rank = rank
    if cfg.dist == 'torch':
        # NOTE launched with init_cuda()
        if cfg.gpu:
            # single node rank -> local GPU index
            cfg.gpu = [cfg.gpu[rank]]
            logging.info(f"[{rank}/{cfg.world_size}]({dist.hostname()}) {utils.get_num_threads()} cores) w/ CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            logging.info(f"[{rank}/{cfg.world_size}]({dist.hostname()}) {utils.get_num_threads()} cores)")
    elif cfg.dist == 'slurm':
        # NOTE launched w/o CUDA initialization yet
        assert cfg.gpu is None
        assert 'CUDA_VISIBLE_DEVICES' in os.environ
        if os.environ['CUDA_VISIBLE_DEVICES'] == 'NoDevFiles':
            logging.info(f"[{rank}/{cfg.world_size}]({dist.hostname()}/{dist.slurm_master()} w/ {utils.get_num_threads()} cores)")
        else:
            # global rank -> local visible GPU(s) instead of absolute SLURM_JOB_GPUS
            devices = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
            cfg.gpu = [devices[cfg.rank % cfg.slurm_ntasks_per_node]]
            logging.info(f"[{rank}/{cfg.world_size}]({dist.hostname()}/{dist.slurm_master()} w/ {utils.get_num_threads()} cores) CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    exec(main, cfg, *args, **kwargs)

def run(main, cfg, *args, **kwargs):
    if cfg.dist:
        if cfg.rank < 0:
            # Launch distributed workers to assign ranks
            if cfg.dist == "torch":
                # spawn one proc per GPU over available locally
                assert cfg.world_size < 0, f"PyTorch distributed world is subject to available local GPUs"
                init_cuda(cfg)
                cfg.world_size = len(cfg.gpu)
                os.environ['MASTER_ADDR'] = dist.hostname()
                os.environ['MASTER_PORT'] = str(cfg.dist_port)
                os.environ['WORLD_SIZE'] = str(cfg.world_size)
                return mp.start_processes(launch, args=(main, cfg, args, kwargs), nprocs=cfg.world_size, daemon=False, join=True, start_method='fork')
            elif cfg.dist == "slurm":
                if 'SLURM_PROCID' not in os.environ:
                    # first time to sbatch with specified resource allocation
                    return dist.slurm_sbatch(cfg)
                else:
                    # launched by SLURM on some (GPU) node
                    dist.slurm_init(cfg, *args, **kwargs)
                    return launch(cfg.rank, main, cfg, args, kwargs)
            else:
                raise ValueError(f"Unsupported distributed mode: {cfg.dist}")
        else:
            assert False, f"Rank must not be set manually"
    else:
        # Local worker with cfg.gpu
        exec(main, cfg, *args, **kwargs)