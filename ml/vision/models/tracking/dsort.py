"""
Adapted from https://github.com/ZQPei/deep_sort_pytorch.
"""

import torch as th
import numpy as np
from .... import logging
from ...ops import *
from .tracking import Tracker, Track

from .deep_sort.nn_matching import NearestNeighborDistanceMetric as NNDistMetric
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker as DSTracker

class DeepSort(Tracker):
    def __init__(self,  max_iou_dist=0.7,   # 0.7
                        max_feat_dist=0.2,  # TODO in case of occlusion?
                        n_init=3,           # 3
                        max_age=13,         # 30 
                        nn_budget=100, 
                        ):
        """DeepSort Tracker wrapper.
        Args:
            max_feat_dist(float): max feature vector distance to considered a match
            max_iou_dist(float): max IoU overlap
            nn_budget(int): if not None, fix samples per class to at most this number. 
                            Removes the oldest samples when the budget is reached.
            max_age(int): maximum number of missed misses before a track is deleted
            n_init(int): number of consecutive detections before the track is confirmed. 
        """
        super().__init__()
        self.tracker = DSTracker(NNDistMetric("cosine", max_feat_dist, nn_budget), 
                                    max_iou_distance=max_iou_dist, 
                                    max_age=max_age, 
                                    n_init=n_init)
    
    @property
    def hits(self):
        return self.tracker.n_init
    
    '''
    def trace(self, tid, history=10):
        track = self.tracks[tid]
        box = torch.from_numpy(track.to_tlwh())
        return xywh2xyxy(box)
    '''

    def update(self, xyxysc, features):
        """Track one time detection.
        Args:
            xyxysc(Tensor[N, 6]): clipped boxes in xyxysc
            features(Tensor[N, D]): pooled RoI features
            frame(Tensoor[C, H, W]): frame to detect
            size(int or Tuple[H, W]): image size to clip boxes
        """
        xyxysc = xyxysc.cpu()
        features = features.cpu().numpy()
        xywh = xyxy2xywh(xyxysc[:, :4]).numpy()
        scores = xyxysc[:, 4]
        classes = xyxysc[:, 5]
        detections = [Detection(xywh[i], score.item(), features[i]) for i, score in enumerate(scores)]

        # Update tracker
        self.tracker.predict()
        matches = self.tracker.update(detections) # { tid: di } including tentative tracks
        if True:
            # Update active tracks with latest detections
            current = set(self.tracks.keys())
            confirmed = { trk.track_id:trk for trk in self.tracker.tracks if trk.is_confirmed() }
            remove = current - set(confirmed.keys())
            for tid in remove:
                del self.tracks[tid]
            tracks = self.tracks
            '''
            print('all tracks:', [trk.track_id for trk in self.tracker.tracks])
            print('current:', sorted(current))
            print('confirmed:', sorted(confirmed.keys()))
            print('remove:', sorted(remove))
            print('matches:', sorted(matches.items()))
            '''
            for tid, trk in confirmed.items():
                if tid in matches:
                    det = xyxysc[matches[tid]]
                    if tid not in tracks:
                        tracks[tid] = Track(tid, cls=int(det[-1].item()))
                    tracks[tid].update(det)
                else:
                    # Not detected => use predicted mean
                    tracked =  xywh2xyxy(th.from_numpy(trk.to_tlwh()).float())
                    tracks[tid].update(tracked)
                    logging.info(f"track[{tid}] not detected since update for {trk.time_since_update} time(s)")
        else:
            # output bbox identities
            outputs = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = xywh2xyxy(box)
                track_id = track.track_id
                outputs.append([x1, y1, x2, y2, track_id])
            if len(outputs) > 0:
                outputs = th.stack(outputs)
            return outputs