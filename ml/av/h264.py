from enum import IntEnum
from ml import logging

__all__ = [
    'NALU_t',
    'NALUParser',
    'H264Framer',
]

THREE_ZEROS = bytes.fromhex('000000')       # Should be replaced with 000003 by encoder
START_CODE24 = bytes.fromhex('000001')
START_CODE32 = bytes.fromhex('00000001')

class NALU_t(IntEnum):
    NIDR = 1 
    IDR = 5
    SEI = 6
    SPS = 7
    PPS = 8
    AUD = 9 # Dahua

def parseNALUHeader(header):
    forbidden = (header & 0x80) >> 7
    ref_idc = (header & 0x60) >> 5
    type = (header & 0x1F)
    return forbidden, ref_idc, type

def NALUParser(bitstream, workaround=False):
    '''
    Args:
        bitstream(bytes-like): a writable bytes-like object that incurs zero copy for slicing
        workaround(bool): removing trailing zero bytes in non-VCL NALUs
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
                count += 1
                header = bitstream[start+start24or32]
                trailing0 = 0
                if workaround:
                    # FIXME Three consecutive zero bytes may be rejected by e.g. KVS
                    # - encoder simply inserts an additionoal zero byte
                    # - encoder forgot to set the stop bit to 1
                    # - encoder forgot to insert an additional byte with MSB set to 1
                    while bitstream[pos - 1 - trailing0] == 0x00:
                        trailing0 += 1
                        logging.warning(f"Skip NALU trailing zero byte at pos {pos - 1 - trailing0}")
                yield (start, *parseNALUHeader(header)), bitstream[start:pos - trailing0]
            start24or32 = 24 // 8
            start = pos
            pos += start24or32
            header = bitstream[pos]
            _, _, type = parseNALUHeader(header)
            if type in (NALU_t.NIDR, NALU_t.IDR):
                # XXX Skip to the end for speedup
                pos = len(bitstream)
        elif next32 == START_CODE32:
            if start < pos:
                header = bitstream[start + start24or32]
                trailing0 = 0
                if workaround:
                    # FIXME Three consecutive zero bytes may be rejected by e.g. KVS
                    # - encoder simply inserts an additionoal zero byte
                    # - encoder forgot to set the stop bit to 1
                    # - encoder forgot to insert an additional byte with MSB set to 1
                    while bitstream[pos - 1 - trailing0] == 0x00:
                        trailing0 += 1
                        logging.warning(f"Skip NALU trailing zero byte at pos {pos - 1 - trailing0}")
                yield (start, *parseNALUHeader(header)), bitstream[start:pos - trailing0]
            start24or32 = 32 // 8
            start = pos
            pos += start24or32
            header = bitstream[pos]
            _, _, type = parseNALUHeader(header)
            if type in (NALU_t.NIDR, NALU_t.IDR):
                # XXX Skip to the end for speedup
                pos = len(bitstream)
        else:
            pos += 1

    # Last NALU to the end
    if start < pos:
        header = bitstream[start+start24or32]
        forbidden, ref_idc, type = parseNALUHeader(header)
        if workaround and type not in (NALU_t.IDR, NALU_t.NIDR):
            # FIXME Three consecutive zero bytes may be rejected by e.g. KVS
            # - encoder simply inserts an additionoal zero byte
            # - encoder forgot to set the stop bit to 1
            # - encoder forgot to insert an additional byte with MSB set to 1
            trailing0 = 0
            while bitstream[pos - 1 - trailing0] == 0x00:
                trailing0 += 1
                logging.warning(f"Skip NALU trailing zero byte at pos {pos - 1 - trailing0}")
            yield (start, forbidden, ref_idc, type), bitstream[start:pos - trailing0]
        else:
            yield (start, forbidden, ref_idc, type), bitstream[start:pos]

# FIXME deprecated since some NALUs must be filtered out or trimmed in place
class H264Framer(object):
    def __init__(self, bitstream):
        '''
        Args:
            bitstream: an object that implements the bitstreamfer protocol
        '''
        super(H264Framer, self).__init__()
        self.bitstream = memoryview(bitstream)
        self.parser = NALUParser(self.bitstream)

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