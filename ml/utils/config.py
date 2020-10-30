from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, IO
import re
import os
import sys
import pprint

import yaml
try:
    from yaml import CLoader as YAMLLoader, CDumper as Dumper
except ImportError:
    from yaml import YAMLLoader, Dumper

class Loader(YAMLLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def _include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')
    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

yaml.add_constructor('!include', _include, Loader)

# XXX Accept scientific notation without decimal point in case of YAML 1.1
yaml.resolver.Resolver.add_implicit_resolver(
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

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self):
        return pprint.pformat(self.__dict__, indent=4)

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
            if isinstance(value, Mapping):
                if self[key] is None:
                    self[key] = Config(value)
                else:
                    self[key].update(value)
            else:
                self[key] = value
        return self

    def clear(self):
        return self.__dict__.clear()

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

        # XXX workaround to merge with defaults
        if 'import' in self:
            imports = self['import']
            for k, v in imports.items():
                if k in self:
                    v.update(self[k])
                if k != 'defaults':
                    self[k] = v
            # Defaults on top level
            if 'defaults' in imports:
                del self['import']
                cfg = imports.defaults
                cfg.update(self)
                self.clear()
                self.update(cfg)    
        return self

    def dump(self):
        return yaml.dump(self.__dict__, Dumper=Dumper, default_flow_style=False)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.dump())
        return
