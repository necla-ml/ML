from pathlib import Path

import pytest
import yaml

from ml.utils import Config

@pytest.fixture
def cfg():
    return Config({
        1: 2,
        'name': 'python',
        'log': {'priority': 1},
    }, key=[1, 'bb', 4])

@pytest.mark.essential
def test_repr(cfg):
    repr = cfg.__repr__()
    str  = cfg.__str__()
    assert eval(repr) == cfg

@pytest.mark.essential
def test_save_load(cfg, tmpdir):
    import tempfile as tf
    with tf.TemporaryDirectory() as tmp:
        filename = Path(tmp, "tmp.yaml")
        cfg.save(filename)
        tmp = Config().load(filename)
        print(filename)
        assert tmp == cfg

@pytest.mark.essential
def test_multiple_option_args_assignment(cfg):
    import tempfile as tf
    with tf.TemporaryDirectory() as tmp:
        filename = Path(tmp, "tmp.yaml")
        cfg.save(filename)
        print(f"cfg={cfg}")

        from ml import argparse
        parser = argparse.ArgumentParser()
        args = parser.parse_args(['--cfg', str(filename), 'log.priority=2', 'key=1e9', 'L=[a,1,c,1e4]', 'name.number=0.34'])
        print(f"args={args}")
        
        for k, v in cfg.items():
            if k not in ['log', 'key', 'name']:
                assert args[k] == v, f"args[{k}] != {v} but {args[k]}"
        assert args.log.priority == 2
        assert args.key == 1e9
        assert args.L == ['a', 1, 'c', 1e4]
        assert args.name.number == 0.34