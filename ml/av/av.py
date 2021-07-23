'''Built on top of pyAV.
'''
from enum import Enum

from ml import sys
from ml import logging

from .backend.pyav import *

'''CODECS available in pyAV/FFMPEG
g711: ['pcm_alaw', 'pcm_mulaw']
g722: ['g722']
g726: ['g726']
amr: ['amrnb', 'amrwb']
aac: ['aac_latm', 'aac', 'aac_at', 'aac_fixed']
vcodecs: ['mjpeg', 'mpeg4', 'h264', 'libx264', 'libopenh264', 'hevc']
'''

class VIDEO_IO_FLAGS(Enum):
    CAP_PROP_POS_MSEC=0
    CAP_PROP_FPS=1
    CAP_PROP_FRAME_HEIGHT=2
    CAP_PROP_FRAME_WIDTH=3
    CAP_PROP_FOURCC=4
    CAP_PROP_BUFFERSIZE=5

RESOLUTIONS = {
    'D1_NTSC': (704, 480),
    'D1_PAL': (704, 576),
    '480p': (640, 480),
    '720p': (1280, 720),
    '960p': (1280, 960),
    '1080p': (1920, 1080),
    '1440p': (2560, 1440),
    '1536p': (2048, 1536),
    '2160p': (3840, 2160),
}

def resolution_str(*res):
    '''Return resolution in WxH as the video_size option.
    Args:
        res(Tuple[int, int] | List[int]): a tuple of width and height or key to RESOLUTIONS
    Returns:
        resolution(str): a string of WxH as an ffmpeg option
    '''
    if len(res) > 1:
        return 'x'.join(map(str, res))
    elif len(res) == 1:
        return resolution_str(*RESOLUTIONS[res[0]])
    else:
        raise ValueError(f"{res} must be of (width, height) or in {list(RESOLUTIONS.keys())}")

def fourcc(fmt):
    if len(fmt) != 4:
        raise ValueError(f"Expected four CC but got {fmt} of {len(fmt)} CC")
    
    if sys.byteorder == 'little':
        ch0, ch1, ch2, ch3 = map(ord, fmt)
    else:
        ch3, ch2, ch1, ch0 = map(ord, fmt)
    return ch3 << 24 | ch2 << 16 | ch1 << 8 | ch0

def fourcc_str(cc):
    cc = int(cc)
    if sys.byteorder == 'little':
        return "".join([chr((cc >> 8 * i) & 0xFF) for i in range(4)])
    else:
        return "".join([chr((cc >> 8 * i) & 0xFF) for i in reversed(range(4))])

CODECS = dict(
    pcm_alaw=fourcc('ALAW'),
    pcm_mulaw=fourcc('ULAW'),
    g722=fourcc('G722'),
    g726=fourcc('G726'),
    amrnb=fourcc('SAMR'),
    amrwb=fourcc('SAWB'),
    aac=fourcc('SAWB'),
    mjpeg=fourcc('MJPG'),
    mpeg4=fourcc('MP42'),
    h264=fourcc('H264'),
    libx264=fourcc('X264'),
    hevc=fourcc('HEVC'),
    yuyv=fourcc('YUYV'),
)


def codec(fmt):
    '''Returns registered codec name in FFMPEG and fourCC.
    
    Args:
        fmt: informal codec format
    
    Returns:
        codec: registered codec name in FFMPEG
        fourcc: four CC of the codec
    '''

    lower = fmt.lower()
    if 'avc1' in lower or '264' in lower:
        return 'h264', CODECS['h264']
    elif 'hevc' in lower or '265' in lower:
        return 'hevc', CODECS['h265']
    elif 'mpeg4' in lower or 'mp42' in lower:
        return 'mpeg4', CODECS['mpeg4']
    elif 'jpg' in lower or 'jpeg' in lower:
        return 'mjpeg', CODECS['mjpeg'] 
    elif 'alaw' in lower:
        return 'pcm_alaw', CODECS['pcm_alaw']
    elif 'ulaw' in lower:
        return 'pcm_mulaw', CODECS['pcm_mulaw'] 
    elif 'yuyv' in lower:
        return 'yuyv', CODECS['yuyv'] 
    else:
        logging.warning(f"Unknown codec format: {fmt}")
        return None, None
