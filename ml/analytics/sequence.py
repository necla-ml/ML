import re
from ml import logging

PATTERNS = dict(
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
    id2chr = list(srange(0x0080, 0x0080 + len(id2cls)))
    return dict(id2cls=id2cls, cls2id=cls2id, id2chr=id2chr)

class SequenceRuleEngine(object):
    def __init__(self, labels, delimiter='->'):
        codebook = encode(labels)
        self.cls2id = codebook['cls2id']
        self.id2chr = codebook['id2chr']
        self.id2cls = codebook['id2cls']
        self.delimiter = delimiter
    
    def parse(self, stage):
        stage = stage.strip().lower()
        
        def cls2chr(m):
            cls = m.group(0)
            if cls == 'anything':
                return f"[{self.id2chr[0]}-{self.id2chr[-1]}]"
            else:
                return self.id2chr[self.cls2id[cls]]

        def repeat(m):
            return f"{{{m.group(1)},}}"

        stage = re.sub(r'[_a-zA-Z]+', cls2chr, stage)
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