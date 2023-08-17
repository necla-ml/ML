import re
import shlex

from ml import logging
from ml.utils import Config

PATTERNS = dict(
    label=re.compile(r"(\"\w[\s\w]*\"|\'\w[\s\w]*\'|[a-zA-Z_]\w*)"),
    repetition=re.compile(r'>=(\d+)')
)

def srange(*args):
    start, step = 0, 1
    if len(args) == 1:
        stop = args[0]
    elif len(args) == 2:
        start, stop = args
    else:
        start, stop, step = args
    start = isinstance(start, str) and ord(start) or start
    stop = isinstance(stop, str) and ord(stop) or stop
    return map(chr, range(start, stop, step))

def encode(labels):
    id2cls = list(labels)
    cls2id = { cls: i for i, cls in enumerate(id2cls) }
    # NOTE: range less than 100 could cause conflicts with python unicodes for whitespaces
    id2chr = list(srange(0x0100, 0x0100 + len(id2cls)))
    return dict(id2cls=id2cls, cls2id=cls2id, id2chr=id2chr)

class SequenceRuleEngine(object):
    def __init__(self, labels, delimiter='->'):
        assert 'anything' not in labels, f"'anything' is already a reserved label"
        codebook = encode(labels)
        self.cls2id = codebook['cls2id']
        self.id2chr = codebook['id2chr']
        self.id2cls = codebook['id2cls']
        self.delimiter = delimiter
        logging.debug(f"Rule codebook\n{Config({label: self.id2chr[self.cls2id[label]] for label in labels})}")
    
    def parse(self, stage):
        stage = stage.strip().lower()
        def repeat(m):
            return f"{{{m.group(1)},}}"

        """
        label must be a quoted string or 
        """
        def cls2chr(m):
            cls = m.group(0).strip("\"\' ")
            if cls == 'anything':
                return f"[{self.id2chr[0]}-{self.id2chr[-1]}]"
            else:
                # logging.info(m, cls)
                return self.id2chr[self.cls2id[cls]]
        # logging.info(f"stage: {stage}")
        stage = re.sub(PATTERNS['label'], cls2chr, stage)
        stage = re.sub(r'\s+', '', stage)
        return re.sub(PATTERNS['repetition'], repeat, stage)

    def compile(self, rule, ending=False):
        stages = rule.split(self.delimiter)
        parsed = ''.join(self.parse(s) for s in stages)
        if ending:
            return re.compile(parsed + '$')
        else:
            return re.compile(parsed)

    def encode(self, *sequence):
        output = []
        for id_or_label in sequence:
            id = id_or_label if isinstance(id_or_label, int) else self.cls2id[id_or_label]
            output.append(self.id2chr[id])
        return ''.join(output)