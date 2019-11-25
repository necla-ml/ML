import os
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from ml import cv

# Thread-safe image dataset with cache
# Metadata: cls, subdir, basename

"""
def pil_imread(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def acc_imread(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_imread(path)

def imread(path, nc=3):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return acc_imread(path)
    else:
        return pil_imread(path)
"""

class ImageDataset(data.Dataset):
    def scan(self, ext):
        if not self.classes:
            # d: None   => nothing to load
            # d: []     => all subdirs
            # d: ['sub1', 'sub2', ...]
            self.classes = { cls.name: [] for cls in self.path.iterdir() if cls.is_dir() and not cls.name.startswith('.') }
        self.targets = sorted(self.classes.keys())
        self.cls2idx = { self.targets[i]: i for i in range(len(self.targets)) }

        for cls in self.targets:
            subdirs = self.classes[cls]
            if subdirs is None: continue
            path = self.path / cls
            if len(subdirs) == 0:
                subdirs.append('.')
                subdirs += [d.name for d in path.iterdir() if d.is_dir()]
            
            #subdirs.sort()
            for subdir in subdirs:
                self.samples += [(cls, subdir, fn.name) for fn in sorted((path / subdir).iterdir()) if fn.suffix in ext]

    def __init__(self, path, classes=None, shuffle=False, input_trans=transforms.Compose([transforms.ToTensor()]), target_trans=None, ext=['.png', '.jpg', '.jpeg'], **kwargs):
        self.path           = Path(path)
        self.classes        = classes
        self.input_trans    = input_trans
        self.target_trans   = target_trans
        self.samples        = []
        self.cache          = {}
        self.cv             = mp.Condition(mp.RLock())
        self.scan(ext)
        self.shuffled       = torch.randperm(len(self.samples)).tolist() if shuffle else list(range(len(self.samples)))
        
    def __getitem__(self, index):
        index = self.shuffled[index]
        with self.cv:
            if index not in self.cache:
                self.cache[index] = False
            else:
                if self.cache[index] is False:
                    while(self.cache[index] is False):
                        print(f'[P{os.getpid()}] waiting for samples[{index}] to be loaded')
                        self.cv.wait()
                    print(f'[P{os.getpid()}] done waiting for samples[{index}]')
        
        cls, subdir, fn = self.samples[index]
        if self.cache[index] is False:
            img = cv.imread(self.path / cls / subdir / fn)
            if self.input_trans:
                img = self.input_trans(img)
            self.cache[index] = img
            with self.cv:
                self.cv.notify_all()

        target = self.cls2idx[cls]
        if self.target_trans:
            target = self.target_trans(target)

        return self.cache[index], target

    def __len__(self):
        return len(self.samples)

class FilterDataset(data.Dataset):
    def __init__(self, src, input_trans=None, target_trans=None, **kwargs):
        self.src = src
        self.input_trans = input_trans
        self.target_trans = target_trans
        
    def __getitem__(self, index):
        src, target = self.src[index]
        if self.input_trans:
            src = self.input_trans(src)
        if self.target_trans:
            target = self.target_trans(target)
        return src, target

    def __len__(self):
        return len(self.src)

class PatchDataset(data.Dataset):
    def __init__(self, src, size, stride=4, input_trans=None, target_trans=None, **kwargs):
        src0 = src[0][0]
        self.src = src
        self.size = size
        self.stride = stride
        self.input_trans = input_trans
        self.target_trans = target_trans
        if torch.is_tensor(src0):
            self.nc, self.height, self.width = src0.size()
        else:
            # CV2: numpy
            self.height, self.width, self.nc = src0.shape

        self.rows = (self.height - size) // stride + 1
        self.cols = (self.width - size) // stride + 1
        self.patches = self.rows * self.cols
        #print(f'size=({self.nc}, {self.height}, {self.width}), rows={self.rows}, cols={self.cols}, patches={self.patches}')

    def __getitem__(self, index):
        if isinstance(index, tuple) and type(index[0] is int):
            z, r, c = index
            #print(f'PatchDataset[{z},{r},{c}]')
            r = r * self.stride
            c = c * self.stride
            img, target = self.src[z]
            #print('region shape:', img.shape)
            img = img[r:r+self.size, c:c+self.size,:]
            #print('patch shape:', img.shape)
            patch = img
            if self.input_trans:
                # XXX One or more inputs pre context
                patch = self.input_trans(img)
            if self.target_trans:
                target = self.target_trans(img)
            
            r'''
            from torchvision.utils import make_grid
            tgt = torch.zeros(patch.size())
            offset = (self.size - 32) // 2
            print(offset, tgt.size())
            tgt[:,offset:2*offset,offset:2*offset].copy_(target)
            grid = make_grid([cv.toTorch(img), patch, tgt], 1)
            cv.show(grid)
            '''
            return patch, target

        elif type(index) is int: # int => tuple
            z = index // self.patches
            p = index - self.patches * z
            r = p // self.cols
            c = p - self.cols * r
            #print(f'[P{os.getpid()}] patch [{index} => ({z},{r},{c})]')
            return self[z, r, c]

    def __len__(self):
        return self.patches * len(self.src)

class SliceDataset(data.Dataset):
    def __init__(self, src, offset=0, size=0):
        offset = offset if offset >= 0 else len(src) + offset
        size = len(src) - offset if size == 0 else size
        assert(offset >= 0 and offset < len(src))
        assert(size >= 0 and size <= len(src) - offset)
        self.src = src
        self.offset = offset
        self.size = size

    def __getitem__(self, index):
        index = index if index >= 0 else self.size + index
        if index < self.size:
            return self.src[self.offset + index]
        raise IndexError(f'index {index} out of range {self.size}')

    def __len__(self):
        return self.size
