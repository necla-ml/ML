from pathlib import Path

def path(filename, mkdir=False):
    r"""Return Path object and create parent directories if specified.
    """
    from pathlib import Path
    filename = Path(filename)
    if mkdir:
        filename.parent.mkdir(parents=True, exist_ok=True)
    return filename
