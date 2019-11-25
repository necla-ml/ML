import re
import sys
import pprint
from pathlib import Path
from importlib import import_module

try:
    import yaml
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# XXX Accept scientific notation without decimal point in case of YAML 1.1
Loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

class Config(dict):
    def __init__(self, args=None, **kwargs):
        # access to keys as to attribues
        super(Config, self).__init__()
        if args is None:
            args = {}
        elif hasattr(args, '__dict__'):
            args = vars(args)
        
        args.update(kwargs)
        self.update(args)

    def __eq__(self, c):
        return type(c) == type(self) and self.__dict__ == c.__dict__ or False

    def __bool__(self):
        return len(self) > 0

    def __contains__(self, key):
        return key in self.__dict__
 
    def __getattr__(self, key):
        return self[key]
    
    def __getitem__(self, key):
        return self.__dict__.get(key) if key in self else None
 
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self, *args):
        return self.__dict__.get(*args)

    def update(self, other=None):
        for key, value in other.items():
            if isinstance(value, dict):
                self[key] = Config(value)
            else:
                self[key] = value
        
        return self

    def load(self, path):
        path = Path(path)
        if path.suffix == '.py':
            module = path.stem
            if '.' in module:
                raise ValueError('Dots are not allowed in config file path')
            sys.path.insert(0, str(path.parent))
            module = import_module(module)
            sys.path.pop(0)
            self.update({
                name: value
                for name, value in module.__dict__.items()
                if not name.startswith('__')
            })
        elif path.suffix in ['.yml', '.yaml', '.json']:
            with open(path, 'r') as f:
                self.update(yaml.load(f, Loader=Loader))
        else:
            raise ValueError(f'Unsupported config file format: {path.suffix}')
            

        return self

    def dump(self):
        return yaml.dump(self.__dict__, Dumper=Dumper, default_flow_style=False)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.dump())
        return
