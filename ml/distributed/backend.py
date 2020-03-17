import os
import sys
import subprocess

from torch import distributed as dist
import torch

from ml import logging


def hostname():
    import socket
    return os.environ.get('SLURMD_NODENAME', socket.gethostname())


def slurm_master(nodes=None):
    #nodes = nodes or os.environ.get('SLURM_NODELIST', None)
    nodes = nodes or os.environ.get('SLURM_JOB_NODELIST', None)
    exitcode, master = subprocess.getstatusoutput(f"scontrol show hostname {nodes} | head -n{1}")
    assert exitcode == 0, f"Failed to find master out of {nodes}"
    return master


def slurm_sbatch(cfg, **kwargs):
    from __main__ import __file__ as script
    from pathlib import Path
    script = Path(script).resolve()
    ntasks = cfg.slurm_ntasks_per_node * cfg.slurm_nodes
    cmd = [
        'srun',
        #f"--mem={cfg.slurm_mem}",
        #f"--job-name={kwargs.get('name', script.name)}",
        #f"--partition={cfg.slurm_partition}",
        #f"--gres=gpu:{cfg.slurm_ntasks_per_node}",          # 8->2
        #f"--ntasks-per-node={cfg.slurm_ntasks_per_node}",   # 8->2
        #f"--ntasks={ntasks}",                               # 8->2
        #f"--cpus-per-task={cfg.slurm_cpus_per_task}",       # 5->3
        #f"--export=ALL,PYTHONPATH=.",
        f"--output=slurm-%j_%t.out",
        '--open-mode=append',
        '--kill-on-bad-exit=1',
        #'-C GPUMODEL_1050TI',                              # scavenger
        #'-C GPUMODEL_1080TI',                             
        #'-C GPUMODEL_RTX2070',
        #'-C GPUMODEL_RTX2080TI',
        #'-C GPUMODEL_TITANX',
        #f"-C {cfg.slurm_constraint}",
        str(script),
    ] + sys.argv[1:]
    cmd = ' '.join(cmd)
    job = f"""#!/bin/sh

#SBATCH --job-name={kwargs.get('name', script.name)}
#SBATCH --partition={cfg.slurm_partition}
#SBATCH --gres=gpu:{0 if cfg.no_gpu else cfg.slurm_ntasks_per_node}       
#SBATCH --ntasks-per-node={cfg.slurm_ntasks_per_node}
#SBATCH --ntasks={ntasks}                            
#SBATCH --cpus-per-task={cfg.slurm_cpus_per_task}
#SBATCH --mem={cfg.slurm_mem}
#SBATCH --time={cfg.slurm_time}
#SBATCH --export=ALL,PYTHONPATH=.
##SBATCH --output=/dev/null
#SBATCH --open-mode=append
{cfg.slurm_constraint and f'#SBATCH -C {cfg.slurm_constraint}' or ''}

{cmd}"""

    sbatch = subprocess.Popen('sbatch', stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        out, err = sbatch.communicate(input=job, timeout=15)
        logging.info(out[:-1])
        print(job)
    except TimeoutExpired:
        sbatch.kill()
        out, err = sbatch.communicate()
        logging.error(f"Timeout to submit the job: {out}, {err}")


def slurm_init(cfg, **kwargs):
    os.environ['MASTER_ADDR'] = slurm_master()
    os.environ['MASTER_PORT'] = str(cfg.dist_port)
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    cfg.world_size = int(os.environ['WORLD_SIZE'])
    cfg.rank = int(os.environ['RANK'])


def mpi_init(backend, **kwargs):
    raise NotImplementedError
