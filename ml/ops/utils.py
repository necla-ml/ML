import torch

def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def boxes2rois(boxes):
    r"""List of bounding box tensors to a ROI tensor with 0th column specifying batch index.

    NOTE:
        Even if no RoI is detected, an empty tensor must be given to decide the device and dimensions to be transparent for RoI pooling.
    """
    dev = boxes[0].device if boxes else None
    concat_boxes = _cat([b for b in boxes], dim=0)
    ids = _cat(
        [
            torch.full_like(b[:, :1], i) # if len(b) > 0 else torch.tensor([], device=dev)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois
