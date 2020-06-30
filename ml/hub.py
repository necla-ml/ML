from torch.hub import *
from ml import logging

ENV_ML_HOME = 'ML_HOME'

def _get_ml_home():
    return os.path.expanduser(
            os.getenv(ENV_ML_HOME,
                      os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                             DEFAULT_CACHE_DIR), 'ml')))

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

# download_url_to_file(url, dst, hash_prefix=None, progress=True):
'''
def download(url, path, force=False):
    path = Path(path)
    res = 0
    if not path.exists() or force:
        cmd = f"curl -LJfq {url} -o {path}"
        logging.info(f"Downloading {url}...")
        logging.info(cmd)
        res = os.system(cmd)
    logging.info(f"Downloaded {url} to {path}")
    return res == 0 and os.path.exists(path)
'''
