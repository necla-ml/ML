'''Built on top of pyAV.
'''

from av import *
from ml import cv
from ml import logging

'''CODECS available in pyAV/FFMPEG
g711: ['pcm_alaw', 'pcm_mulaw']
g722: ['g722']
g726: ['g726']
amr: ['amrnb', 'amrwb']
aac: ['aac_latm', 'aac', 'aac_at', 'aac_fixed']
vcodecs: ['mjpeg', 'mpeg4', 'h264', 'libx264', 'libopenh264', 'hevc']
'''


def fourcc(fmt):
    if len(fmt) != 4:
        raise ValueError(f"Expected four CC but got {fmt} of {len(fmt)} CC")
    return cv.VideoWriter_fourcc(*fmt)    


def fourcc_str(cc):
    cc = int(cc)
    return "".join([chr((cc >> 8 * i) & 0xFF) for i in range(4)])


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


def avcodec(fmt):
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
