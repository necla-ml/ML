from pathlib import Path

import pytest
import yaml

from ml.utils import Config


@pytest.fixture
def cfg():
    return Config({
        1: 2,
        'name': 'python',
        'logging': {'priority': 1},
    }, key=[1, 'bb', 4])#, logging={'level': 4})


def test_repr(cfg):
    repr = cfg.__repr__()
    str  = cfg.__str__()
    assert eval(repr) == cfg


def test_save_load(cfg, tmpdir):
    import tempfile as tf
    with tf.TemporaryDirectory() as tmp:
        filename = Path(tmpdir, "tmp.yaml")
        cfg.save(filename)
        tmp = Config().load(filename)
        print(filename)
        assert tmp == cfg