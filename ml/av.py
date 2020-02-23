'''Built on top of pyAV.
'''

from av import *
from ml import cv
from ml import logging


def avcodec(fmt):
    fmt = fmt.replace('.', '')
    fmt = fmt.replace('jpeg', 'jpg')
    upper = fmt.upper()
    lower = fmt.lower()
    if upper in ['YUYV', 'MJPG', 'H264', 'X264', 'H265', 'HEVC']:
        return upper, cv.VideoWriter_fourcc(*upper)
    elif 'avc1' in lower or '264' in lower:
        return 'H264', cv.VideoWriter_fourcc(*'H264')
    elif 'hevc' in lower or '265' in lower:
        return 'H265', cv.VideoWriter_fourcc(*'H265')
    elif 'alaw' in lower:
        return 'pcm_alaw', cv.VideoWriter_fourcc(*'alaw')
    elif 'mulaw' in lower:
        return 'pcm_mulaw', cv.VideoWriter_fourcc(*'ulaw') 
    else:
        logging.warning(f"Unknown codec format: {fmt}")
        return fmt, None