from pathlib import Path
import argparse
import re

from .utils.config import Config
from ml import logging

class ConfigAction(argparse.Action):
    def __inint__(self, opt, dest, help='Specify an external config to load options', **kwargs):
        super(ConfigAction, self).__init__(opt, 'cfg', help=help, **kwargs)

    def __call__(self, parser, args, values, opt):
        r"""Parse and combine command line options with a configuration file if specified.
        """
        # print(f"values={values}")
        for value in values:
            if '=' in value:
                # assignement
                k, v = value.split('=')
                parts = k.split('.')
                key = parts[-1]
                parts = parts[:-1]
                # print(f"assignment: '{value}', parts={parts}, key={key}")
                cfg = vars(args)
                # print('cfg:', cfg)
                for part in parts:
                    if part not in cfg or not isinstance(cfg[part], Config):
                        cfg[part] = Config()
                    cfg = cfg[part]
                
                import yaml
                v = yaml.safe_load(v)
                cfg[key] = v
            else:
                # path to config
                path = Path(value)
                if not path.exists():
                    from importlib import import_module
                    path = Path(import_module(parser.__module__).__file__).parent / "configs" / path.name
                    logging.info(f"Loading by default from {path}")
                cfg = Config()
                cfg.load(path)
                for k, v in vars(cfg).items():
                    field = args.__dict__.get(k)
                    if isinstance(field, Config):
                        field.update(v)
                    else:
                        args.__dict__[k] = v

class ArgumentParser(argparse.ArgumentParser):
    r"""Allow no '=' in an argument config as a regular command line
    """
    
    def __init__(self, *args, **kwargs):
        char = kwargs.get("fromfile_prefix_chars", '@')
        kwargs["fromfile_prefix_chars"] = char
        super(ArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument("--cfg", action=ConfigAction, nargs='+', help="path to a configuration file in YAML/JSON or a Python module")
        
        # App/Logging
        self.add_argument('-d', '--daemon', action='store_true',                                                help='Daemonize to detach from the shell')
        self.add_argument('-L', '--logging', action='store_true',                                               help='Enable logging')
        self.add_argument('--logfile', default=None,                                                            help='Path to log messages if provided')

        # CUDA
        self.add_argument('-s','--seed', default='1204', type=int,                                                help='Random seed')
        self.add_argument('--deterministic', action='store_true',                                               help='Deterministic training results')
        self.add_argument('--gpu', nargs='+', type=int,                                                         help='One or more visible GPU indices as CUDA_VISIBLE_DEVICES w/o comma')
        self.add_argument('--no-gpu', action='store_true',                                                      help='Only use CPU regardless of --gpu')

        # Distributed
        self.add_argument('--dist', nargs='?', default=None, const='torch', choices=['torch', 'slurm'],         help='Enable distribued launch')
        self.add_argument('--dist-backend', default='nccl', choices=['nccl', 'gloo', 'mpi'],                    help='Distributed backend')
        self.add_argument('--dist-url', default='env://',                                                       help='URL to set up distributed training')
        self.add_argument('--dist-port', default=25900, type=int,                                               help='Port to set up distributed training')
        self.add_argument('--dist-no-sync-bn', action='store_true',                                             help='Disable synchronized batch normalization')
        
        self.add_argument('--world-size', default=-1, type=int,                                                 help='number of nodes for distributed training')
        self.add_argument('--rank', default=-1, type=int,                                                       help='Node rank for distributed training')
        
        self.add_argument('--slurm-partition', default='gpu',                                                   help='Slurm job partition to submit to')
        self.add_argument('--slurm-nodes', default=1, type=int,                                                 help='Number of worker nodes to launch job')
        self.add_argument('--slurm-ntasks-per-node', default=1, type=int,                                       help='Number of GPU procs to launch per node')
        self.add_argument('--slurm-cpus-per-task', default=4, type=int,                                         help='Number of CPUs per proc')
        self.add_argument('--slurm-constraint', 
                          # default='GPUMODEL_RTX2080TI', 
                          choices=['GPUMODEL_TITANX', 'GPUMODEL_1080TI', 'GPUMODEL_RTX2080TI'], 
                          help='GPU model constraint w.r.t. memory capacity')
        self.add_argument('--slurm-mem', default='16G',                                                         help='Max host memory to allocate')
        self.add_argument('--slurm-time', default='0',                                                          help='Time limit')
        self.add_argument('--slurm-export', default='',                                                         help='Time limit')

    def convert_arg_line_to_args(self, line):
        tokens = line.strip().split()
        first = re.split(r"=", tokens[0])
        return first + tokens[1:]

    def parse_args(self, args=None, ns=None):
        args = super(ArgumentParser, self).parse_args(args, ns)
        cfg = Config(args)
        return cfg

    def parse_known_args(self, args=None, ns=None):
        args, leftover = super(ArgumentParser, self).parse_known_args(args, ns)
        cfg = Config(args)
        del cfg.cfg
        return cfg, leftover
