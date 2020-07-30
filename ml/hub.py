from pathlib import Path
from time import time
from torch.hub import *
from ml import logging

ENV_ML_HOME = 'ML_HOME'

def _get_ml_home():
    return os.path.expanduser(
            os.getenv(ENV_ML_HOME,
                      os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                             DEFAULT_CACHE_DIR), 'ml')))

def parse(url):
    if url.startswith('s3://'):
        parts = url[len('s3://'):].split('/')
        key = '/'.join(parts[1:])
        return dict(
            bucket=parts[0],
            key=key,
            name=parts[-1],
        )
    else:
        return None

def github(owner, project, tag=None):
    if tag is None:
        return f"{owner}/{project}"
    else:
        return f"{owner}/{project}:{tag}"

def repo(github, force_reload=False, verbose=True):
    # pytorch-1.5.1 applicable
    from torch.hub import _get_cache_or_reload
    import os
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

def download_s3(bucket, key, path='.', force=False):
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
        path = Path(path)
        if path.is_dir():
            path /= Path(key).name
        if path.is_file():
            if force:
                path.unlink()
                logging.warning(f"Removed existing download: {path}")
            else:
                logging.warning(f"Download exists: {path}, specify force=True to remove if necessary")
                return True
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
            s3.download_file(bucket, key, str(path), Callback=callback)
        except ClientError as e:
            print()
            logging.error(f"Failed to download s3://{bucket}/{key} to {path}: {e}")
            return False
        else:
            print()
            logging.info(f"Succeeded to download s3://{bucket}/{key} to {path}")
            return True

def load_state_dict_from_s3(bucket, key, filename, model_dir=None, map_location=None, force_reload=False, progress=True, check_hash=False):
    if model_dir is None:
        # Otherwise, hub_dir would be used.
        ml_home = _get_ml_home()
        model_dir = os.path.join(ml_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    path = f"{model_dir}/{filename}"
    if not download_s3(bucket, key, path, force=force_reload):
        raise IOError(f"Failed to download to {path}")

    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    cached_file = path
    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)
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
    if model_dir is None:
        # Otherwise, hub_dir would be used.
        ml_home = _get_ml_home()
        model_dir = os.path.join(ml_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    path = f"{model_dir}/{filename}"
    if download_gdrive(id, path, force=force_reload) != 0:
        raise IOError(f"Failed to download to {path}")

    cached_file = path
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