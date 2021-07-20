from typing import Union
from ml import logging

try:
    from ml.vision.io import *
except Exception as e:
    logging.warn(f"{e}, ml.vision.io unavailable, run `mamba install ml-vision -c NECLA-ML` to install")
else:
    from pathlib import Path
    from ml.vision import io
    import torch as th

    def load(path: Union[str, Path], *args):
        r"""Load a file in supported image or video formats.

        Other args and kwargs from APIs to support:
        - read_image(path: str, mode: torchvision.io.image.ImageReadMode = <ImageReadMode.UNCHANGED: 0>) → torch.Tensor
        - read_video(filename: str, start_pts: int = 0, end_pts: Optional[float] = None, pts_unit: str = "pts") -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]
        """
        suffix = Path(path).suffix.lower() 
        if suffix in ['.png', '.jpg', '.jpeg']:
            return io.read_image(str(path), *args)
        else:
            # Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]
            return io.read_video(str(path), *args)


    def save(data: th.Tensor, path: Union[str, Path], *args, **kwargs):
        """Encode image to save in jpeg or png.
        Args:
            data (PIL.Image, accimage.Image, Tensor[C, H, W, dtype=uint8] or Tensor[T, H, W, C, dtype=uint8]))

        Other args and kwargs from APIs to support:
        - write_jpeg(input: torch.Tensor, filename: str, quality: int = 75)
        - write_png(input: torch.Tensor, filename: str, compression_level: int = 6)
        - write_video(filename: str, 
                    video_array: torch.Tensor, 
                    fps: float, 
                    video_codec: str = 'libx264', 
                    options: Optional[Dict[str, Any]] = None, 
                    audio_array: Optional[torch.Tensor] = None, 
                    audio_fps: Optional[float] = None, 
                    audio_codec: Optional[str] = None, 
                    audio_options: Optional[Dict[str, Any]] = None)

        NOTE:
            No longer support opencv
        """
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        from ..transforms import functional as F
        if suffix in ('.jpg', '.jpeg'):
            if not F.is_tensor(data):
                data = F.to_tensor(data)
            io.write_jpeg(data, str(path), *args)
        elif path.suffix.lower() == '.png':
            if not F.is_tensor(data):
                data = F.to_tensor(data)
            io.write_png(data, str(path), *args)
        else:
            io.write_video(str(path), data, *args, **kwargs)