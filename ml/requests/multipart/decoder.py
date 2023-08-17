import io
import requests, base64
from requests import Request, Session
from requests_toolbelt import MultipartDecoder
from requests_toolbelt.multipart.decoder import *

from ml import logging

__all__ = [
    'HTTPError',            # HTTP related
    'HTTPStatus',
    'HTTPRequest',
    'HTTPRequestStreamDecoder',
    'MultipartStreamDecoder',
    'RequestException',     # Anything else
    'Timeout',              # ConnectionTimeout, ReadTime
]

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
    def from_response(cls, resp, encoding='utf-8'):
        content = resp.raw
        content_type = resp.headers.get('content-type', None)
        return cls(content, content_type, encoding)

    def _parse_body(self, content):
        '''
        boundary = b''.join((b'--', self.boundary))
        def body_part(part):
            fixed = MultipartDecoder._fix_first_part(part, boundary)
            return BodyPart(fixed, self.encoding)

        def test_part(part):
            return (part != b'' and
                    part != b'\r\n' and
                    part[:4] != b'--\r\n' and
                    part != b'--')
        '''
        # TODO Support any bsize
        self.parts = PartIterator(content, self.boundary, newline=b'\r\n', bsize=96)
    
    def __iter__(self):
        return self.parts

from requests.exceptions import RequestException, HTTPError, Timeout
from http.client import HTTPException
from http import HTTPStatus
import http

class HTTPRequest(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class HTTPRequestParser(object):
    r"""Parser for a continuous stream of HTTP requests.
    Adapted from http.server and http.client.
    """

    default_request_version = "HTTP/0.9"
    protocol_version = "HTTP/1.1"

    def __init__(self, rfile):
        self.rfile = rfile

    def __next__(self):
        if hasattr(self, 'close_connection') and self.close_connection:
            logging.warning(f"Connection closed")
            raise StopIteration
        else:
            req = self.parse()
            if req is None:
                logging.warning(f"No requests")
                raise StopIteration
            return req

    def parse(self):
        while True:
            self.raw_requestline = self.rfile.readline(65537)
            if len(self.raw_requestline) > 65536:
                raise BufferError(HTTPStatus.REQUEST_URI_TOO_LONG)
            if not self.raw_requestline:
                self.close_connection = True
                return
            if self.raw_requestline == b'\r\n':
                continue
            break
        logging.debug(f"request: {self.raw_requestline}")
        return self.parse_one()

    def parse_one(self):
        """Parse a request (internal).
        Return a parsed HTTPRequest instance for success, None for failure.
        """

        self.close_connection = True
        command = None  # set in case of error on the first line
        version = self.default_request_version
        requestline = str(self.raw_requestline, 'iso-8859-1')
        requestline = requestline.rstrip('\r\n')
        words = requestline.split()
        if len(words) == 0:
            return None

        if len(words) >= 3:  # Enough to determine protocol version
            version = words[-1]
            try:
                if not version.startswith('HTTP/'):
                    raise ValueError
                base_version_number = version.split('/', 1)[1]
                version_number = base_version_number.split(".")
                # RFC 2145 section 3.1 says there can be only one "." and
                #   - major and minor numbers MUST be treated as
                #      separate integers;
                #   - HTTP/2.4 is a lower version than HTTP/2.13, which in
                #      turn is lower than HTTP/12.3;
                #   - Leading zeros MUST be ignored by recipients.
                if len(version_number) != 2:
                    raise ValueError
                version_number = int(version_number[0]), int(version_number[1])
            except (ValueError, IndexError):
                raise HTTPError(HTTPStatus.BAD_REQUEST, f"Bad request version ({version}) in request: {requestline}")
            if version_number >= (1, 1) and self.protocol_version >= "HTTP/1.1":
                self.close_connection = False
            if version_number >= (2, 0):
                raise HTTPError(f"Invalid HTTP version ({base_version_number})")

        if not 2 <= len(words) <= 3:
            raise HTTPError(f"Bad request syntax ({requestline})")
            
        method, path = words[:2]
        if len(words) == 2:
            self.close_connection = True
            if method != 'GET':
                raise HTTPError(f"Bad HTTP/0.9 request type ({method})")

        # Examine the headers and look for a Connection directive.
        try:
            headers = http.client.parse_headers(self.rfile)
        except http.client.LineTooLong as err:
            raise HTTPError(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE, "Line too long", str(err))
        except http.client.HTTPException as err:
            raise HTTPError(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE, "Too many headers", str(err))

        conntype = headers.get('Connection', "")
        if conntype.lower() == 'close':
            self.close_connection = True
        elif (conntype.lower() == 'keep-alive' and self.protocol_version >= "HTTP/1.1"):
            self.close_connection = False
        
        # Examine the headers and look for an Expect directive
        expect = headers.get('Expect', "")
        if (expect.lower() == "100-continue" and 
                self.protocol_version >= "HTTP/1.1" and
                version >= "HTTP/1.1"):
            raise NotImplementedError("Unsupported HTTP Expect header")

        size = int(headers.get('Content-Length', 0))
        body = self.rfile.read(size)
        self.rfile.readline()
        return HTTPRequest(
            method=method,
            path=path,
            version=version,
            headers=headers,
            body=body
        )

class HTTPRequestStreamDecoder(MultipartDecoder):
    r"""Decode continuous HTTP requests.
    Assume the communication is set to the stream mode.
    """

    @classmethod
    def from_response(cls, resp, encoding='utf-8'):
        content_type = resp.headers.get('content-type', None)
        return cls(resp.raw, content_type, encoding)

    def _find_boundary(self):
        return

    def _parse_body(self, rfile):
        self.parts = HTTPRequestParser(rfile)
    
    def __iter__(self):
        return self.parts