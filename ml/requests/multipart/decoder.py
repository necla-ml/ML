import requests, base64
from requests import Request, Session
from requests_toolbelt import MultipartDecoder
from requests_toolbelt.multipart.decoder import *

from ml import io, logging

# Suppress unnecessary warnings from urllib3
class NoHeaderErrorFilter(logging.Filter):
    """Filter out urllib3 Header Parsing Errors due to a urllib3 bug."""

    def filter(self, record):
        """Filter out Header Parsing Errors."""
        return "Failed to parse headers" not in record.getMessage()


def filter_urllib3_logging():
    """Filter header errors from urllib3 due to a urllib3 bug."""
    urllib3_logger = logging.getLogger("urllib3.connectionpool")
    if not any(isinstance(x, NoHeaderErrorFilter) for x in urllib3_logger.filters):
        urllib3_logger.addFilter(NoHeaderErrorFilter())


filter_urllib3_logging()


class PartIterator(object):
    def __init__(self, stream, boundary, newline=b'\r\n', encoding='utf-8', bsize=96):
        self.stream = io.BufferedReader(stream)
        self.boundary = b''.join((b'--', boundary, newline))
        self.newline = newline
        self.encoding = encoding
        self.buf = bytearray()
        self.bsize = bsize

    def __next__(self):
        boundary = self.stream.read(len(self.boundary))
        #print('##### boundary: ', boundary)
        assert boundary == self.boundary, f'Unexpected boundary: {boundary}'
        self.buf += self.stream.read(self.bsize)
        if len(self.buf) < self.bsize:
            logging.error(f'EOS: only {len(buf)} bytes in the buffer < {self.bsize}')
            raise StopIteration

        #print(f'peek: {buf[:self.bsize]}')
        part = BodyPart(self.buf, self.encoding)
        if b'content-length' in part.headers:
            length = int(part.headers[b'content-length'].decode(self.encoding)) # excluding newline
            # assert length >= self.bsize, f"{length} < bsize({self.bsize})"
            size = length - len(part.content) # + len(self.newline)
            if size < 0: # over read
                # print(f"[PartIterator] Over read {-size} bytes")
                self.buf = part.content[size:]
                part.content = part.content[:size]
                if len(self.buf) < len(self.newline):
                    self.stream.read(len(self.newline) - len(self.buf))
                    self.buf.clear()
                else:
                    self.buf = self.buf[len(self.newline):]
            else: # under read
                part.content += self.stream.read(size)
                self.stream.read(len(self.newline))
                self.buf.clear()
            # print('[PartIterator]', part.headers, len(part.content))
            return part
        else:
            logging.error(f'No Content-Length in the part headers: {part.headers}')
            raise StopIteration


class MultipartStreamDecoder(MultipartDecoder):
    @classmethod
    def from_response(cls, response, encoding='utf-8'):
        content = response.raw
        content_type = response.headers.get('content-type', None)
        return cls(content, content_type, encoding)

    def _parse_body(self, content):
        boundary = b''.join((b'--', self.boundary))

        def body_part(part):
            fixed = MultipartDecoder._fix_first_part(part, boundary)
            return BodyPart(fixed, self.encoding)

        def test_part(part):
            return (part != b'' and
                    part != b'\r\n' and
                    part[:4] != b'--\r\n' and
                    part != b'--')
        
        # TODO Support any bsize
        self.parts = PartIterator(content, self.boundary, newline=b'\r\n', bsize=96)
    
    def __iter__(self):
        return self.parts