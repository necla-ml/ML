from pathlib import Path
from time import time
from torch.hub import *
from ml import logging

'''
ENV_ML_HOME = 'ML_HOME'
def _get_ml_home():
    return os.path.expanduser(
            os.getenv(ENV_ML_HOME,
                      os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                             DEFAULT_CACHE_DIR), 'ml')))
'''

SPECS = dict(
    s3='s3://',
    github='https://github.com/'
)

def parse(url):
    if url.startswith(SPECS['s3']):
        parts = url[len(SPECS['s3']):].split('/')
        key = '/'.join(parts[1:])
        return dict(
            bucket=parts[0],
            key=key,
            name=parts[-1],
        )
    elif url.startswith(SPECS['github']):
        # https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5x.pt
        parts = url[len(SPECS['github']):].split('/')
        owner, project, tag, name = parts[0], parts[1], parts[4], parts[5]
        return dict(
            owner=owner,
            project=project, 
            tag=tag,
            name=name
        )
    else:
        logging.warning('Unknown url spec={url}')
        return None

def github(owner, project, tag=None):
    if tag is None:
        return f"{owner}/{project}"
    else:
        return f"{owner}/{project}:{tag}"

def github_release_url(owner, project, tag, name):
    return f"https://github.com/{owner}/{project}/releases/download/{tag}/{name}"

def repo(github, force_reload=False, verbose=True):
    # pytorch-1.6.0 applicable
    from torch.hub import _get_cache_or_reload
    import os
    if True:
        os.makedirs(get_dir(), exist_ok=True)
    else:
        hubdir = f"{_get_ml_home()}/hub"
        os.makedirs(hubdir, exist_ok=True)
        set_dir(hubdir)
    return _get_cache_or_reload(github, force_reload, verbose)

def upload_s3(path, bucket, key):
    '''
    Args:
        path(str): path to the file to upload
        bucket(str): S3 bucket name
        key(str): key to upload to the bucket where the ending '/' matters
    '''
    try:
        import botocore, boto3
        from botocore.exceptions import ClientError
    except ImportError as e:
        logging.warning(f'botocore and boto3 are required to download from S3: {e}')
        return False
    else:
        # XXX Amazon S3 supports buckets and objects, and there is no hierarchy.
        path = Path(path)
        s3 = boto3.resource('s3').meta.client
        if not path.is_file():
            logging.error(f"{path} not exist or not a file to upload")
            return False
        total = 0
        start = time()
        def callback(bytes):
            nonlocal total
            total += bytes
            elapse = time() - start
            if total < 1024:
                print(f"\rUploaded {total:4d} bytes at {total / elapse:.2f} bytes/s", end='')
            elif total < 1024**2:
                KB = total / 1024
                print(f"\rUploaded {KB:4.2f}KB at {KB/elapse:4.2f} KB/s", end='')
            else:
                MB = total / 1024**2
                print(f"\rUploaded {MB:8.2f}MB at {MB/elapse:6.2f} MB/s", end='')
            sys.stdout.flush()
        try:
            print(path, bucket, key)
            s3.upload_file(str(path), bucket, key, Callback=callback)
        except ClientError as e:
            print()
            logging.error(f"Failed to upload {path} to s3://{bucket}/{key}: {e}")
        else:
            print()
            logging.info(f"Succeeded to upload {path} to s3://{bucket}/{key}")
        return True

def download_s3(bucket, key, path=None, progress=True):
    '''
    Args:
        bucket(str): S3 bucket name
        key(str): path to a file to download in the bucket
    Kwargs:
        path(str): directory to save the downloaded file named by the key or the target path to save
    '''
    try:
        import botocore, boto3
        from botocore.exceptions import ClientError
    except ImportError as e:
        logging.warning(f'botocore and boto3 are required to download from S3: {e}')
        return False
    else:
        s3 = boto3.client('s3', config=botocore.client.Config(max_pool_connections=50))
        path = Path(path or '.')
        if path.is_dir():
            path /= Path(key).name
        
        total = 0
        start = time()
        def callback(bytes):
            nonlocal total
            total += bytes
            elapse = time() - start
            if total < 1024:
                print(f"\rDownloaded {total:4d} bytes at {total / elapse:.2f} bytes/s", end='')
            elif total < 1024**2:
                KB = total / 1024
                print(f"\rDownloaded {KB:4.2f}KB at {KB/elapse:4.2f} KB/s", end='')
            else:
                MB = total / 1024**2
                print(f"\rDownloaded {MB:8.2f}MB at {MB/elapse:6.2f} MB/s", end='')
            sys.stdout.flush()
        try:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=path.parent)
            s3.download_file(bucket, key, tmp.name, Callback=progress and callback or None)
        except ClientError as e:
            print()
            logging.error(f"Failed to download s3://{bucket}/{key} to {path}: {e}")
            return False
        else:
            from ml import shutil
            shutil.move(tmp.name, path)
            print()
            logging.info(f"Succeeded to download s3://{bucket}/{key} to {path}")
            return True

def load_state_dict_from_url(url, model_dir=None, map_location=None, force_reload=False, progress=True, check_hash=False, file_name=None):
    # FIXME Temporary workaround for pytorch-1.6.0 introducing new checkpoint save in zip format
    # Added argument: force_reload
    # Alternative url scheme: s3
    r"""Loads the Torch serialized object at the given URL.
    If downloaded file is a zip file, it will be automatically decompressed.
    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set.
    
    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')
    
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    spec = urlparse(url)
    filename = os.path.basename(spec.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    
    download = True
    if os.path.exists(cached_file):
        if force_reload:
            os.unlink(cached_file)
            logging.warning(f"Forced removing existing download: {cached_file}")
        else:
            download = False
            logging.warning(f"Download exists: {cached_file}, specify force_reload=True to remove if necessary")
    
    if download:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        try:
            if spec.scheme == 's3':
                bucket = spec.netloc
                key = spec.path[1:]
                download_s3(bucket, key, cached_file, progress=progress)
            else:
                hash_prefix = None
                if check_hash:
                    r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
                    hash_prefix = r.group(1) if r else None
                download_url_to_file(url, cached_file, hash_prefix, progress=progress)
        except Exception as e:
            raise IOError(f"Failed to download to {cached_file}: {e}")
    '''
    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location)
    '''
    logging.info(f'Loading checkpoint from {cached_file}')
    return torch.load(cached_file, map_location=map_location)

def download_gdrive(id='1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS', path='/tmp/yolov5x.pt', force=False):
    # https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    import time
    t = time.time()

    if os.path.exists(path):
        if force:
            os.remove(path)
            logging.warning(f"Removed existing download: {path}")
        else:
            logging.warning(f"Download exists: {path}, specify force=True to remove if necessary")
            return 0
    
    logging.info(f'Downloading https://drive.google.com/uc?export=download&id={id} to {path}...')
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    os.system(f"curl -c ./cookie -s -L \'https://drive.google.com/uc?export=download&id={id}\' > /dev/null")
    if os.path.exists('cookie'):  # large file
        # s = "curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=%s\" -o %s" % (id, path)
        s = f"curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id={id}\" -o {path}"
    else:  # small file
        s = f"curl -s -L -o {path} 'https://drive.google.com/uc?export=download&id={id}'"
    r = os.system(s)  # execute, capture return values
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(path) if os.path.exists(path) else None  # remove partial
        logging.error(f'Failed to download to {path}')  # raise Exception('Download error')
        return r

    '''
    # Unzip if archive
    if path.endswith('.zip'):
        logging.info('Unzipping... ')
        os.system('unzip -q %s' % path)  # unzip
        os.remove(path)  # remove zip to free space
    '''
    logging.info(f'Done in {time.time() - t:.1f}s')
    return r

def load_state_dict_from_gdrive(id, filename, model_dir=None, map_location=None, force_reload=False, progress=True, check_hash=False):
    """Assume the downloaded file is saved by torch directly through `torch.save()` in zip format since pytorch-1.6.
    """
    '''
    if model_dir is None:
        # Otherwise, hub_dir would be used.
        ml_home = _get_ml_home()
        model_dir = os.path.join(ml_home, 'checkpoints')
    '''
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    cached_file = f"{model_dir}/{filename}"
    if download_gdrive(id, cached_file, force=force_reload) != 0:
        raise IOError(f"Failed to download to {cached_file}")

    '''
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)
    '''
    return torch.load(cached_file, map_location=map_location)    