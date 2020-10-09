from pathlib import Path
from ml import logging

def export(model, args, f, **kwargs):
    path = Path(f)
    if path.suffix == '.onnx':
        from torch import onnx
        logging.info(f"Exporting pytorch model to {f}")
        onnx.export(model, args, str(path), **kwargs)
    else:
        raise ValueError(f"Unknown suffix `{path.suffix}` to export")