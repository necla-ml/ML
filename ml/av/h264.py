from enum import IntEnum
from ml import logging

__all__ = [
    'H264Framer',
]

THREE_ZEROS = bytes.fromhex('000000')       # Should be replaced with 000003 by encoder
START_CODE24 = bytes.fromhex('000001')
START_CODE32 = bytes.fromhex('00000001')

class NALU_t(IntEnum):
    NIDR = 1 
    IDR = 5
    SPS = 7
    PPS = 8

def parseNALUHeader(header):
    forbidden = (header & 0x80) >> 7
    ref_idc = (header & 0x60) >> 5
    type = (header & 0x1F)
    return forbidden, ref_idc, type

def NALUParser(bitstream, workaround=False):
    '''
    Args:
        bitstream: a memory view that incurs zero copy
    '''

    # FIXME Assume at most three NALUs for fast parsing
    pos = 0
    start = 0
    count = 0
    start24or32 = -1
    while pos < len(bitstream):
        next24 = bitstream[pos:pos+3]
        next32 = bitstream[pos:pos+4]
        if next24 == START_CODE24:
            if start < pos:
                if workaround and bitstream[pos - 1] == 0x00:
                    # FIXME Three consecutive zero bytes may be rejected by e.g. KVS
                    # - encoder forgot to set the stop bit to 1
                    # - encoder forgot to insert an additional byte with MSB set to 1
                    bitstream[pos - 1] = 0x01
                    logging.warning(f"In-place mutation to work around three consecutive zeros at pos {pos - 1}")
                count += 1
                header = bitstream[start+start24or32]
                yield (start, *parseNALUHeader(header)), bitstream[start:pos]
            start24or32 = 24 // 8
            start = pos
            pos += start24or32
            header = bitstream[pos]
            _, _, type = parseNALUHeader(header)
            if type in [NALU_t.NIDR | NALU_t.IDR]:
                pos = len(bitstream)
        elif next32 == START_CODE32:
            if start < pos:
                if workaround and bitstream[pos - 1] == 0x00:
                    # FIXME Three consecutive zero bytes may be rejected by e.g. KVS
                    # - encoder forgot to set the stop bit to 1
                    # - encoder forgot to insert an additional byte with MSB set to 1
                    bitstream[pos - 1] = 0x01
                    logging.warning(f"In-place mutation to work around three consecutive zeros at pos {pos - 1}")
                header = bitstream[start + start24or32]
                yield (start, *parseNALUHeader(header)), bitstream[start:pos]
            start24or32 = 32 // 8
            start = pos
            pos += start24or32
            header = bitstream[pos]
            _, _, type = parseNALUHeader(header)
            if type in [NALU_t.NIDR | NALU_t.IDR]:
                pos = len(bitstream)
        else:
            pos += 1

    if start < pos:
        header = bitstream[start+start24or32]
        yield (start, *parseNALUHeader(header)), bitstream[start:pos]

class H264Framer(object):
    def __init__(self, bitstream, workaround=False):
        '''
        Args:
            bitstream: an object that implements the bitstreamfer protocol
        '''
        super(H264Framer, self).__init__()
        self.workaround = workaround
        self.bitstream = memoryview(bitstream)
        self.parser = NALUParser(self.bitstream, workaround)

    def __iter__(self):
        return self

    def __next__(self):
        (pos, forbidden, ref_idc, type), nalu = next(self.parser)
        if type == NALU_t.SPS:
            (pos_pps, _, _, type_pps), pps = next(self.parser)
            (pos_idr, _, _, type_idr), idr = next(self.parser)
            return True, self.bitstream[pos:pos+len(nalu)+len(pps)+len(idr)]
        else:
            return False, self.bitstream[pos:pos+len(nalu)]